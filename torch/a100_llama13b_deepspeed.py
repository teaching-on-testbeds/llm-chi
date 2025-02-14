import torch
import litgpt
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
        super().__init__()
 
        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
            print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters())

batch_size = 4
accumulate_grad_batches=4

lit_model = LitLLM(checkpoint_dir="openlm-research/open_llama_13b")
data = Alpaca2k(val_split_fraction=0.2)  # to save time - use only 80% of the data for training 

data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

trainer = L.Trainer(
    devices=4,
    strategy=DeepSpeedStrategy(
        stage=3,                 # Similar to FULL_SHARD
        offload_optimizer=True   # Enable CPU offloading of optimizer
    ),
    accelerator="gpu",
    max_epochs=1,
    accumulate_grad_batches=accumulate_grad_batches,
    precision="bf16-true",
    limit_val_batches=0,            # to save time - don't bother with validation
    enable_checkpointing = False    # to save time - don't bother saving fine-tuned model

)
trainer.fit(lit_model, data)
