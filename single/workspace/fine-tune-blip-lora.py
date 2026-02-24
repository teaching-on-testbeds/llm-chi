import torch
import pytorch_lightning as L
import time
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_from_disk
import bitsandbytes as bnb
from deepspeed.ops.adam import DeepSpeedCPUAdam
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Configuration - edit these to experiment with different settings
cfg = {
    "model_name": "Salesforce/blip2-opt-2.7b",  # "Salesforce/blip2-opt-2.7b" or "Salesforce/blip2-opt-6.7b" or "Salesforce/blip2-flan-t5-xl" or "Salesforce/blip2-flan-t5-xxl"
    "lr": 5e-6,
    "batch_size": 2,
    "accumulate_grad_batches": 2,
    "precision": "bf16-true",   # or "32-true", "bf16-mixed", "bf16-true", "fp16-true", etc.
    "optim": "adamw", # can use "adamw", "sgd", "adam_8bit", or "deepspeed_cpu"
    "act_ckpt": False, # uses gradient_checkpointing_enable() on single GPU, FSDP policies on multi-GPU
    "max_epochs": 5,
    "max_steps": -1,  # e.g. 50 to stop early, -1 otherwise
    "devices": 1,  # set to >1 for multi-GPU
    "strategy": 'auto',  
    "num_workers": 8,
    "limit_train_batches": 1.0,  # e.g. 50 to debug quickly
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False,
    "use_lora": False,       # True to enable LoRA (freezes base model, trains small adapter)
    "use_qlora": False,      # True to enable QLoRA (4-bit base model + LoRA)
    "lora_r": 16,            # LoRA rank
    "lora_alpha": 32,        # LoRA scaling factor
    "lora_dropout": 0.05,
}

print("[cfg]")
for k in sorted(cfg):
    print(f"  {k}: {cfg[k]}")

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

        # QLoRA: load model in 4-bit quantized format
        if cfg["use_qlora"]:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, quantization_config=bnb_config
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            model_load_dtype = torch.bfloat16 if cfg["precision"] == "bf16-true" else torch.float32
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                dtype=model_load_dtype
            )

        # LoRA / QLoRA: wrap model with LoRA adapters
        if cfg["use_lora"] or cfg["use_qlora"]:
            lora_config = LoraConfig(
                r=cfg["lora_r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=["q", "v"],
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        if cfg["act_ckpt"]:
            self.model.gradient_checkpointing_enable()
        self.model.train()
        self.lr = lr

    def _print_memory_breakdown(self, tag="", log=True):
        model = self.model
        optimizer = self.trainer.optimizers[0]

        param_mem_gpu = sum(p.numel() * p.element_size() for p in model.parameters() if p.is_cuda)
        trainable_param_mem = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

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
        act_mem = allocated - (param_mem_gpu + grad_mem_gpu + opt_mem_gpu)
        act_mem = max(0, act_mem)

        mem_dict = {
            "param_mem_GB" + tag: param_mem_gpu/1024**3,
            "trainable_param_mem_GB" + tag: trainable_param_mem/1024**3,
            "grad_mem_GB" + tag: grad_mem_gpu/1024**3,
            "opt_mem_GB" + tag: opt_mem_gpu/1024**3,
            "opt_mem_cpu_GB" + tag: opt_mem_cpu/1024**3,
            "act_mem_GB" + tag: act_mem/1024**3,
            "allocated_mem_GB" + tag: allocated/1024**3,
            "max_allocated_mem_GB" + tag: allocated_max/1024**3,
        }

        if log:
            self.log_dict(mem_dict, prog_bar=True, on_step=True)

        strategy_cfg = cfg.get("strategy")
        is_deepspeed = isinstance(strategy_cfg, str) and "deepspeed" in strategy_cfg.lower()
        if is_deepspeed:
            print(
                f"[{tag}] Params: {param_mem_gpu/1024**3:.2f} GB (trainable: {trainable_param_mem/1024**3:.2f} GB) | "
                f"Grads: {grad_mem_gpu/1024**3:.2f} GB | "
                f"Optim GPU: {opt_mem_gpu/1024**3:.2f} GB | "
                f"Optim CPU: {opt_mem_cpu/1024**3:.2f} GB | "
                f"Acts: {act_mem/1024**3:.2f} GB | "
                f"Allocated: {allocated/1024**3:.2f} GB (peak {allocated_max/1024**3:.2f}) | "
                f"Reserved: {reserved/1024**3:.2f} GB"
            )
        else:
            print(
                f"[{tag}] Params: {param_mem_gpu/1024**3:.2f} GB (trainable: {trainable_param_mem/1024**3:.2f} GB) | "
                f"Grads: {grad_mem_gpu/1024**3:.2f} GB | "
                f"Optim: {opt_mem_gpu/1024**3:.2f} GB | "
                f"Acts: {act_mem/1024**3:.2f} GB | "
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
        first_param_dtype = next(self.model.parameters()).dtype
        print(f"[info] precision={cfg['precision']} | first parameter dtype={first_param_dtype}")

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
        print(f"[epoch {self.current_epoch}] avg_train_loss: {avg_loss:.4f}")
        self._memory_log.append({"epoch": self.current_epoch, "avg_train_loss": avg_loss, **snapshot})
        self._epoch_losses = []  # reset for next epoch

    def on_train_end(self):
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        elapsed_sec = time.perf_counter() - self._train_start_time
        print(f"\n[done] train time: {elapsed_sec:.1f} s | peak memory: {peak_gb:.2f} GB")

    def configure_optimizers(self):
        # When using LoRA/QLoRA, only optimize trainable (adapter) params
        if cfg["use_lora"] or cfg["use_qlora"]:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
        else:
            params = self.model.parameters()

        if cfg["optim"] == "adamw":
            return torch.optim.AdamW(params, lr=self.lr)
        if cfg["optim"] == "sgd":
            return torch.optim.SGD(params, lr=self.lr)
        if cfg["optim"] == "adam_8bit":
            return bnb.optim.Adam8bit(params, lr=self.lr)
        if cfg["optim"] == "deepspeed_cpu":
            return DeepSpeedCPUAdam(params, lr=self.lr)


# Trainer
model = BLIP2FoodFineTuner(cfg["model_name"], cfg["lr"])
trainer = L.Trainer(
    accelerator="gpu",
    devices=cfg["devices"],
    strategy=cfg["strategy"],
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
