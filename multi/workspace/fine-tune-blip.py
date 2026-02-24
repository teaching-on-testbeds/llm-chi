import torch
import pytorch_lightning as L
import time
import os
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers.models.blip_2.modeling_blip_2 import Blip2EncoderLayer, Blip2QFormerLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from datasets import load_from_disk
import bitsandbytes as bnb

# Configuration
cfg = {
    "model_name": "Salesforce/blip2-opt-2.7b", # "Salesforce/blip2-opt-2.7b" or "Salesforce/blip2-opt-6.7b" or "Salesforce/blip2-flan-t5-xl" or "Salesforce/blip2-flan-t5-xxl"
   
    "lr": 2e-5,
    "batch_size": 16,
    "accumulate_grad_batches": 4,
    "precision": "bf16-true", # or "32-true", "bf16-mixed", "bf16-true", "fp16-true", etc.
    "optim": "adamw", # can use "adamw", "sgd", "adam_8bit"
    "act_ckpt": False, # uses gradient_checkpointing_enable() on single GPU, FSDP policies on multi-GPU
    "max_epochs": 4,
    "max_steps": -1,
    "devices": 1,
    "strategy": "auto",  # e.g. "auto", "ddp", "fsdp"
    "num_workers": 8,
    "limit_train_batches": 1.0,
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False
}

def _is_rank0_process():
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank is not None:
        return int(rank) == 0
    if local_rank is not None:
        return int(local_rank) == 0
    return True


if _is_rank0_process():
    print("[cfg]")
    for k in sorted(cfg):
        print(f"  {k}: {cfg[k]}")

# Build FSDP-aware strategy when needed
_blip2_layer_cls = {Blip2EncoderLayer, Blip2QFormerLayer, OPTDecoderLayer}


def build_strategy(cfg):
    strategy = cfg["strategy"]
    if strategy == "auto":
        return "auto"
    if strategy == "ddp":
        return DDPStrategy(gradient_as_bucket_view=True)
    if isinstance(strategy, str) and strategy.startswith("fsdp"):
        kwargs = {"auto_wrap_policy": _blip2_layer_cls}
        if cfg["act_ckpt"]:
            kwargs["activation_checkpointing_policy"] = _blip2_layer_cls
        return FSDPStrategy(**kwargs)
    return strategy

#  Dataset
dataset = load_from_disk("./data/gourmetgram_caption")
dataset = dataset["train"]
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(min(cfg["num_train_samples"], len(dataset))))
def collate_fn(batch):
    return {
        "image": [x["image"] for x in batch],
        "text": [x["text"] for x in batch],
    }

train_loader = DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=cfg["num_workers"],
    drop_last=True  # ensures that all batches are same size
)


# Model
class BLIP2FoodFineTuner(L.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32 # note: model will be cast to bf16 automatically if we use reduced precision
        )
        if cfg["act_ckpt"] and cfg["strategy"] == 'auto':
            self.model.gradient_checkpointing_enable()
        self.model.train()
        self.lr = lr

    def _print_memory_breakdown(self, tag="", log=True):
        model = self.model
        optimizer = self.trainer.optimizers[0]

        param_mem_gpu = sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda)
        grad_mem_gpu = sum(
            p.grad.numel() * p.grad.element_size()
            for p in model.parameters()
            if p.grad is not None and p.grad.is_cuda
        )
        opt_mem_gpu = sum(
            sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v) and v.is_cuda)
            for state in optimizer.state.values()
        )
        opt_mem_cpu = sum(
            sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v) and not v.is_cuda)
            for state in optimizer.state.values()
        )

        allocated = torch.cuda.memory_allocated()
        allocated_max = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.memory_reserved()
        # In distributed runs, this residual includes activations plus any
        # non-attributed CUDA allocations (for example DDP/FSDP communication
        # buffers and collective buckets).
        act_mem = allocated - (param_mem_gpu + grad_mem_gpu + opt_mem_gpu)
        act_mem = max(0, act_mem)

        mem_dict = {
            "param_mem_GB" + tag: param_mem_gpu/1024**3,
            "grad_mem_GB" + tag: grad_mem_gpu/1024**3,
            "opt_mem_GB" + tag: opt_mem_gpu/1024**3,
            "opt_mem_cpu_GB" + tag: opt_mem_cpu/1024**3,
            "act_mem_GB" + tag: act_mem/1024**3,
            "allocated_mem_GB" + tag: allocated/1024**3,
            "max_allocated_mem_GB" + tag: allocated_max/1024**3,
        }

        if log:
            self.log_dict(mem_dict, prog_bar=True, on_step=True)

        if self.trainer.is_global_zero:
            print(
                f"[{tag}] Params: {param_mem_gpu/1024**3:.2f} GB | "
                f"Grads: {grad_mem_gpu/1024**3:.2f} GB | "
                f"Optim: {opt_mem_gpu/1024**3:.2f} GB | "
                f"Other: {act_mem/1024**3:.2f} GB | "
                f"Allocated: {allocated/1024**3:.2f} GB (peak {allocated_max/1024**3:.2f}) | "
                f"Reserved: {reserved/1024**3:.2f} GB"
            )
        return mem_dict

    def _should_log_step(self):
        """Log at global steps 0, 1 (optimizer init) and every 50 steps thereafter."""
        step = self.trainer.global_step
        return step <= 1 or step % 50 == 0

    def on_train_start(self):
        self.model.train()
        self._train_start_time = time.perf_counter()
        self._memory_log = []  # collect snapshots for saving
        self._epoch_losses = []  # track per-step losses for epoch average

    def on_train_batch_start(self, batch, batch_idx):
        if self._should_log_step():
            self._print_memory_breakdown(tag=f"_step{self.trainer.global_step}_phase0_before_fwd")

    def on_before_backward(self, loss):
        if self._should_log_step():
            self._print_memory_breakdown(tag=f"_step{self.trainer.global_step}_phase1_after_fwd")

    def on_after_backward(self):
        if self._should_log_step():
            self._print_memory_breakdown(tag=f"_step{self.trainer.global_step}_phase2_after_bwd")

    def training_step(self, batch, _):
        vision_inputs = self.processor(images=batch["image"], return_tensors="pt").to(self.device)
        text_inputs = self.processor.tokenizer(
            batch["text"], padding=True, truncation=True, max_length=50, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=vision_inputs["pixel_values"],
            labels=text_inputs["input_ids"],
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self._epoch_losses.append(loss.detach().item())
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        if self._should_log_step():
            self._print_memory_breakdown(tag=f"_step{self.trainer.global_step}_phase3_after_opt_step")

    def on_train_epoch_end(self):
        snapshot = self._print_memory_breakdown(tag=f"_epoch_end_{self.current_epoch}", log=False)
        avg_loss = sum(self._epoch_losses) / len(self._epoch_losses) if self._epoch_losses else 0
        if self.trainer.is_global_zero:
            print(f"[epoch {self.current_epoch}] avg_train_loss: {avg_loss:.4f}")
        self._memory_log.append({"epoch": self.current_epoch, "avg_train_loss": avg_loss, **snapshot})
        self._epoch_losses = []  # reset for next epoch

    def on_train_end(self):
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        elapsed_sec = time.perf_counter() - self._train_start_time
        if self.trainer.is_global_zero:
            print(f"\n[done] train time: {elapsed_sec:.1f} s | peak memory: {peak_gb:.2f} GB")

    def configure_optimizers(self):
        if cfg["optim"] == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        if cfg["optim"] == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if cfg["optim"] == "adam_8bit":
            return bnb.optim.Adam8bit(self.model.parameters(), lr=self.lr)


# Trainer
model = BLIP2FoodFineTuner(cfg["model_name"], cfg["lr"])
trainer = L.Trainer(
    accelerator="gpu",
    devices=cfg["devices"],
    strategy=build_strategy(cfg),
    precision=cfg["precision"],
    accumulate_grad_batches=cfg["accumulate_grad_batches"],
    max_epochs=cfg["max_epochs"],
    max_steps=cfg["max_steps"],
    limit_train_batches=cfg["limit_train_batches"],
    enable_checkpointing=cfg["enable_checkpointing"],
    enable_progress_bar=False,
    log_every_n_steps=1
)
trainer.fit(model, train_loader)

# Save model
if cfg["save_model"] and trainer.is_global_zero:
     save_dir = "blip2-food-checkpoint"
     print(f"Saving model to {save_dir}...")
     model.model.save_pretrained(save_dir)
     model.processor.save_pretrained(save_dir)
     print(f"Model and processor saved to: {save_dir}")
