
::: {.cell .markdown}

## Train a large model on a single GPU 

In this section, we will practice strategies for training a large model on a single GPU. After completing this section, you should understand the effect of

* batch size
* gradient accumulation
* reduced precision/mixed precision
* CPU offload
* activation checkpointing
* parameter efficient fine tuning

on a large model training job.

:::


::: {.cell .markdown}

Make sure that you can see the GPU inside the container:

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
nvidia-smi
```
:::

::: {.cell .markdown}

Throughout these experiments, we will monitor GPU compute and memory utilization with `nvtop`. Open a terminal in the Jupyter service on node-llm-single (File > New > Terminal) and in it, run 

```bash
# runs in the Jupyter service on node-llm-single
nvtop
```

In this display,

* `GPU%` tells us how busy the GPU compute cores are. Low values with long training time often mean the GPU is waiting (for data, CPU work, or synchronization).
* `GPU mem%` tells us how much GPU VRAM is currently in use. If this approaches 100%, we are close to OOM.

We will refer back to this display throughout the experiment.
:::

::: {.cell .markdown}

Before running the training scripts, download and unpack the dataset snapshot that we will use in this lab.

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
cd ~/work
mkdir -p data
wget -O data/gourmetgram_caption.tar.gz "https://nyu.box.com/shared/static/g3qw3g5j7l8dkvyf02a9afuuo9grs3g3.gz"
mkdir -p data/gourmetgram_caption
tar -xzf data/gourmetgram_caption.tar.gz -C data/gourmetgram_caption --strip-components=1

ls -lah data/gourmetgram_caption
```
:::

::: {.cell .markdown}

The training scripts now read the dataset from `./data/gourmetgram_caption`.

:::
::: {.cell .markdown}

### PyTorch Lightning workflow

In this section, we will run a BLIP-2 fine-tuning script (`fine-tune-blip.py`) that is built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

PyTorch Lightning helps us keep the training loop logic mostly fixed while we change strategy using config values like:

* batch size
* gradient accumulation
* precision mode (`32-true`, `bf16-mixed`, `bf16-true`)
* activation checkpointing
* optimizer choice
* LoRA/QLoRA options

This means we can compare memory and training time across many settings by editing one config dictionary in a script, instead of rewriting training code each time.

Our focus will be on comparing time and memory requirements under different settings - we aren't trying to optimize for model quality.

:::


::: {.cell .markdown}

First, let's briefly review the training script, which is written for Pytorch Lightning. Lightning is Pytorch with less boilerplate and some additional functionality baked in, including things like distributed training.

A basic Lightning module for our image captioning model would look something like this:

```python
# Model
class BLIP2FoodFineTuner(L.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float32 
        )
        self.model.train()
        self.lr = lr

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
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

```

But ours is a little more complicated for two reasons: configuration and logging.

At the top, the `cfg = {...}` dictionary controls the experiment settings. Then, those settings are used by the Lightning `Trainer`. Lightning implements the techniques we learned about so we just have to specify them - we don't have to implement them from scratch in Pytorch.


```python
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
```

We also use some of the configuration inside the Lightning module itself, e.g. 

```python
def configure_optimizers(self):
    if cfg["optim"] == "adamw":
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    if cfg["optim"] == "sgd":
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)
    if cfg["optim"] == "adam_8bit":
        return bnb.optim.Adam8bit(self.model.parameters(), lr=self.lr)
    if cfg["optim"] == "deepspeed_cpu":
        return DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr)
```

We have also added some extra logging, so that the script prints memory snapshots at key points in a training step:

* `_phase0_before_fwd`: right before the forward pass
* `_phase1_after_fwd`: after forward, before backward
* `_phase2_after_bwd`: after backward
* `_phase3_after_opt_step`: after optimizer step (when optimizer state is materialized)

Lightning makes it easy for us to define custom code that runs at various points in the forward pass, backward pass, or optimizer step.

Each memory snapshot will include:

* `Params`: model weights in memory
* `Grads`: gradient tensors 
* `Optim`: optimizer state tensors 
* `Acts`: activation memory (Note: we actually don't have a good way to estimate this exactly, so we are assuming "whatever is left" is activation memory usage. In fact, this number as reported also includes other small non-activation items stored in memory, which are cumulatively less than 1GB.)
* `Allocated/peak`: current and max allocated CUDA memory seen so far
* `Reserved`: memory held by the CUDA allocator cache (can stay high even after tensors are freed)

and it will print these in step 0, step 1, and then every 50 steps after that.

:::


::: {.cell .markdown}

Open `fine-tune-blip.py` in the Jupyter file browser. We will use a baseline config first, then edit only a few keys at each experiment stage.

For every experiment in this notebook, we will follow the same run loop:

1. edit the requested keys in `cfg`
2. run `python fine-tune-blip.py` in a terminal cell
3. record whether it succeeds or fails, plus run time and memory output

:::


::: {.cell .markdown .gpu-a100}

For quick reference, this table summarizes the configuration used in each full fine-tuning experiment.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Model</th>
      <th>bs/acc</th>
      <th>precision</th>
      <th>optim</th>
      <th>act_ckpt</th>
      <th>strategy</th>
      <th>max_steps</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Baseline</td><td>blip2-opt-2.7b</td><td>32/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, expected OOM</td></tr>
    <tr><td>Reduced batch size</td><td>blip2-opt-2.7b</td><td>16/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Gradient accumulation</td><td>blip2-opt-2.7b</td><td>16/4</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Grad accum + LR x4 rerun</td><td>blip2-opt-2.7b</td><td>16/4</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=2e-5</td></tr>
    <tr><td>Reduced precision</td><td>blip2-opt-2.7b</td><td>16/4</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Mixed precision</td><td>blip2-opt-2.7b</td><td>16/4</td><td>bf16-mixed</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Larger model</td><td>blip2-opt-6.7b</td><td>16/4</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Even larger model</td><td>blip2-flan-t5-xxl</td><td>16/4</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>expected OOM</td></tr>
    <tr><td>XXL + smallest batch</td><td>blip2-flan-t5-xxl</td><td>1/1</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>expected OOM</td></tr>
    <tr><td>Optimizer without state</td><td>blip2-flan-t5-xxl</td><td>16/4</td><td>bf16-true</td><td>sgd</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>8-bit optimizer</td><td>blip2-flan-t5-xxl</td><td>2/2</td><td>bf16-true</td><td>adam_8bit</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Activation checkpointing</td><td>blip2-flan-t5-xxl</td><td>2/2</td><td>bf16-true</td><td>adam_8bit</td><td>True</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>CPU offload (DeepSpeed)</td><td>blip2-flan-t5-xxl</td><td>16/4</td><td>bf16-true</td><td>deepspeed_cpu</td><td>False</td><td>deepspeed_stage_2_offload</td><td>2</td><td></td></tr>
  </tbody>
</table>

And this table summarizes the PEFT experiments.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Model</th>
      <th>bs/acc</th>
      <th>precision</th>
      <th>optim</th>
      <th>act_ckpt</th>
      <th>strategy</th>
      <th>max_steps</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LoRA</td><td>blip2-flan-t5-xxl</td><td>32/2</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, use_lora=True, use_qlora=False</td></tr>
    <tr><td>QLoRA</td><td>blip2-flan-t5-xxl</td><td>64/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, use_lora=False, use_qlora=True</td></tr>
  </tbody>
</table>

:::


::: {.cell .markdown .gpu-h100}

For quick reference, this table summarizes the configuration used in each full fine-tuning experiment.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Model</th>
      <th>bs/acc</th>
      <th>precision</th>
      <th>optim</th>
      <th>act_ckpt</th>
      <th>strategy</th>
      <th>max_steps</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Baseline</td><td>blip2-opt-2.7b</td><td>64/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, OOM</td></tr>
    <tr><td>Reduced batch size</td><td>blip2-opt-2.7b</td><td>16/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Gradient accumulation</td><td>blip2-opt-2.7b</td><td>16/4</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Grad accum + LR x4 rerun</td><td>blip2-opt-2.7b</td><td>16/4</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=2e-5</td></tr>
    <tr><td>Reduced precision</td><td>blip2-opt-2.7b</td><td>16/4</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Mixed precision</td><td>blip2-opt-2.7b</td><td>16/4</td><td>bf16-mixed</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Larger model</td><td>blip2-opt-6.7b</td><td>32/2</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Even larger model</td><td>blip2-flan-t5-xxl</td><td>32/2</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>OOM</td></tr>
    <tr><td>XXL + smallest batch</td><td>blip2-flan-t5-xxl</td><td>1/1</td><td>bf16-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>OOM</td></tr>
    <tr><td>Optimizer without state</td><td>blip2-flan-t5-xxl</td><td>32/2</td><td>bf16-true</td><td>sgd</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>8-bit optimizer</td><td>blip2-flan-t5-xxl</td><td>2/2</td><td>bf16-true</td><td>adam_8bit</td><td>False</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>Activation checkpointing</td><td>blip2-flan-t5-xxl</td><td>2/2</td><td>bf16-true</td><td>adam_8bit</td><td>True</td><td>auto</td><td>-1</td><td></td></tr>
    <tr><td>CPU offload (DeepSpeed)</td><td>blip2-flan-t5-xxl</td><td>32/2</td><td>bf16-true</td><td>deepspeed_cpu</td><td>False</td><td>deepspeed_stage_2_offload</td><td>2</td><td>set num_workers=0 to reduce host RAM pressure</td></tr>
  </tbody>
</table>

And this table summarizes the PEFT experiments.

<table>
  <thead>
    <tr>
      <th>Experiment</th>
      <th>Model</th>
      <th>bs/acc</th>
      <th>precision</th>
      <th>optim</th>
      <th>act_ckpt</th>
      <th>strategy</th>
      <th>max_steps</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LoRA</td><td>blip2-flan-t5-xxl</td><td>32/2</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, use_lora=True, use_qlora=False</td></tr>
    <tr><td>QLoRA</td><td>blip2-flan-t5-xxl</td><td>64/1</td><td>32-true</td><td>adamw</td><td>False</td><td>auto</td><td>-1</td><td>lr=5e-6, num_train_samples=512, use_lora=False, use_qlora=True</td></tr>
  </tbody>
</table>


:::


::: {.cell .markdown}

### Experiment: Baseline

As a baseline, let's try two epochs of fine-tuning Blip-2 using OPT-2.7b, an LLM with 2.7 billion parameters. When using Blip-2 with OPT-2.7b, the combined model will have 3.7 billion parameters.

:::

::: {.cell .markdown .gpu-a100}

Set `cfg` in `fine-tune-blip.py` to this baseline:

```python
cfg = {
    "model_name": "Salesforce/blip2-opt-2.7b",
    "lr": 5e-6,
    "batch_size": 32,
    "accumulate_grad_batches": 1,
    "precision": "32-true",
    "optim": "adamw",
    "act_ckpt": False,
    "max_epochs": 2,
    "max_steps": -1,
    "devices": 1,
    "strategy": "auto",
    "num_workers": 8,
    "limit_train_batches": 1.0,
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False,
}
```

:::

::: {.cell .markdown .gpu-h100}

Set `cfg` in `fine-tune-blip.py` to this baseline:

```python
cfg = {
    "model_name": "Salesforce/blip2-opt-2.7b",
    "lr": 5e-6,
    "batch_size": 64,
    "accumulate_grad_batches": 1,
    "precision": "32-true",
    "optim": "adamw",
    "act_ckpt": False,
    "max_epochs": 2,
    "max_steps": -1,
    "devices": 1,
    "strategy": "auto",
    "num_workers": 8,
    "limit_train_batches": 1.0,
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False,
}
```

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

This run is expected to fail with OOM. 

In the logs, we should see that parameters alone take close to 14 GB at full precision (32-bit). Then, 

* After the forward pass, activation memory spikes. 
* After backward pass, activation memory is freed, but now gradients are saved in memory. 
* After the optimizer step, optimizer state is allocated, memory jumps again, and the run goes OOM.

:::


::: {.cell .markdown}

### Experiment: Reduced batch size

What if we reduce the batch size?

:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"batch_size": 32` -> `"batch_size": 16`

Leave all other values the same as the baseline.

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"batch_size": 64` -> `"batch_size": 16`

Leave all other values the same as the baseline.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

This run should now fit in memory, because with a smaller batch size, we need substantially less memory for activations. 

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Experiment: Gradient accumulation

By using gradient accumulation to "step" only after a few "micro batches", we can train with a larger effective "global" batch size, with minimal effect on the memory required.

:::

::: {.cell .markdown}

In `cfg`, change:

* `"accumulate_grad_batches": 1` -> `"accumulate_grad_batches": 4`

Keep `batch_size` at `16` and leave all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

With gradient accumulation, we will see more output lines per step, simply because there are multiple forward passes and backward passes before the optimizer step.

Make a note of the training time and memory. We should see memory similar to the previous run.

:::

::: {.cell .markdown}

You may notice that the loss after two epochs is different from the previous run - since the effective batch size has changed, we should scale the learning rate by the same amount to have a similar total step size over an epoch. 

In the `cfg`, increase the learning rate by 4x (`"lr": 5e-6` -> `"lr": 2e-5`), and run again:

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::



::: {.cell .markdown}

### Experiment: Reduced precision

With a "bf16" format for numbers, instead of "float32", we can further reduce the memory required, although this representation is less precise.

:::

::: {.cell .markdown}

In `cfg`, change:

* `"precision": "32-true"` -> `"precision": "bf16-true"`

Keep all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

This job should run much faster, and use much less memory, but it may also have higher loss than the previous run.

:::





::: {.cell .markdown}

### Experiment: Mixed precision

With mixed precision, we get back some of the lost precision in the results, at the cost of additional memory.

:::

::: {.cell .markdown}

In `cfg`, change:

* `"precision": "bf16-true"` -> `"precision": "bf16-mixed"`

Keep all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

You may notice that this job is faster than the equivalent full precision job, with comparable loss after two epochs.

:::

::: {.cell .markdown}

### Experiment: Larger model 

We've gained so much GPU memory back with these techniques, we can even train a larger model. Let's switch to a larger BLIP-2 model while keeping `bf16-true` precision:
:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-opt-2.7b"` -> `"Salesforce/blip2-opt-6.7b"`
* `"batch_size": 16` (keep at 16)
* `"accumulate_grad_batches": 4` (keep at 4)
* `"precision": "bf16-mixed"` -> `"bf16-true"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-opt-2.7b"` -> `"Salesforce/blip2-opt-6.7b"`
* `"batch_size": 16` -> `"batch_size": 32`
* `"accumulate_grad_batches": 4` -> `"accumulate_grad_batches": 2`
* `"precision": "bf16-mixed"` -> `"bf16-true"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Experiment: Even larger model

Let us try an even larger model. We will change out the LLM from `opt-6.7b` (6.7B parameters) to `flan-t5-xxl` (about 11B parameters).

:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-opt-6.7b"` -> `"Salesforce/blip2-flan-t5-xxl"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-opt-6.7b"` -> `"Salesforce/blip2-flan-t5-xxl"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

This run is expected to fail with OOM. 

:::


::: {.cell .markdown}

### Experiment: Even larger model with even smaller batch size

Even if we reduce to the smallest possible batch size, the `flan-t5-xxl` model is still too large for us to train on this GPU.

:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"batch_size": 16` -> `"batch_size": 1`
* `"accumulate_grad_batches": 4` -> `"accumulate_grad_batches": 1`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"batch_size": 32` -> `"batch_size": 1`
* `"accumulate_grad_batches": 2` -> `"accumulate_grad_batches": 1`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

This run will OOM, too.

:::




::: {.cell .markdown}

### Experiment: Optimizer without state (SGD)

The previous XXL runs fail partly because optimizer state takes a lot of memory. Next, we keep the XXL model but switch to an optimizer with no state.

:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"batch_size": 1` -> `"batch_size": 16`
* `"accumulate_grad_batches": 1` -> `"accumulate_grad_batches": 4`
* `"optim": "adamw"` -> `"optim": "sgd"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"batch_size": 1` -> `"batch_size": 32`
* `"accumulate_grad_batches": 1` -> `"accumulate_grad_batches": 2`
* `"optim": "adamw"` -> `"optim": "sgd"`

Leave all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of training time and memory. Compare `Optim` memory with the previous run.

While we have freed up enough memory to train the XXL model, the result may not be great - the loss after 2 epochs may be poor.

:::

::: {.cell .markdown}

### Experiment: Low-memory optimizer alternative (`adam_8bit`)

Another option is 8-bit Adam, which keeps optimizer state in reduced precision.

:::

::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"optim": "sgd"` -> `"optim": "adam_8bit"`
* `"batch_size": 16` -> `"batch_size": 2`
* `"accumulate_grad_batches": 4` -> `"accumulate_grad_batches": 2`

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"optim": "sgd"` -> `"optim": "adam_8bit"`
* `"batch_size": 32` -> `"batch_size": 2`

:::

::: {.cell .markdown}

Keep all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of training time and memory.

:::

::: {.cell .markdown}

### Experiment: Activation checkpointing

Another way to reduce memory usage (at the cost of compute) is with activation checkpointing. 

:::


::: {.cell .markdown}

In `cfg`, change:

* `"act_ckpt": False` -> `"act_ckpt": True`

:::


::: {.cell .markdown}

Keep all other values the same as the previous experiment.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of training time and memory, especially the activation memory.

We will see that we have saved on the activation memory, but at the cost of much slower training.

:::

::: {.cell .markdown}

### Experiment: CPU offload with DeepSpeed

Finally, another way to save memory at the cost of compute is to use DeepSpeed CPU offload to move optimizer state off the GPU.

:::


::: {.cell .markdown .gpu-a100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-flan-t5-xxl"`
* `"precision": "bf16-true"`
* `"optim": "deepspeed_cpu"`
* `"act_ckpt": False`
* `"strategy": "deepspeed_stage_2_offload"`
* `"batch_size": 16`
* `"accumulate_grad_batches": 4`
* `"max_steps": 2`

:::

::: {.cell .markdown .gpu-h100}

In `cfg`, change:

* `"model_name": "Salesforce/blip2-flan-t5-xxl"`
* `"precision": "bf16-true"`
* `"optim": "deepspeed_cpu"`
* `"act_ckpt": False`
* `"strategy": "deepspeed_stage_2_offload"`
* `"batch_size": 32`
* `"accumulate_grad_batches": 2`
* `"num_workers": 0`
* `"max_steps": 2`

:::

::: {.cell .markdown}

This resets us to the "Even larger model" settings and then adds CPU offload. "Stage 2" here refers to ZeRO stage 2 - offloading optimizer state and gradients.

We set the maximum number of steps to 2 in this case, because training with CPU offload will be *so* slow - we really don't want to let it run to the end of 2 epochs.

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Make a note of training time and memory. We should see lower GPU memory pressure, with extra overhead from CPU offload.

In this run, the memory breakdown prints optimizer memory separately for GPU (`Optim GPU`) and CPU (`Optim CPU`). With DeepSpeed offload, a large fraction of optimizer state should move to CPU, so we expect `Optim CPU` to be high while `Optim GPU` stays much smaller.

:::

::: {.cell .markdown}

### Experiment: Parameter efficient fine tuning

If we are only fine-tuning, not training a model from scratch, we can also consider LoRA and QLoRA. Let's try it first with our XXL model.

We are going to use the `fine-tune-blip-lora.py` script for our PEFT experiments, so open that and note

:::

::: {.cell .markdown}

Set `cfg` in `fine-tune-blip-lora.py` to this LoRA configuration:

```python
cfg = {
    "model_name": "Salesforce/blip2-flan-t5-xxl",
    "lr": 5e-6,
    "batch_size": 32,
    "accumulate_grad_batches": 2,
    "precision": "32-true",
    "optim": "adamw",
    "act_ckpt": False,
    "max_epochs": 2,
    "max_steps": -1,
    "devices": 1,
    "strategy": "auto",
    "num_workers": 8,
    "limit_train_batches": 1.0,
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False,
    "use_lora": True,
    "use_qlora": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
```

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference). Also check the `[cfg]` output near the top of the run to confirm the script is using the config you intended.

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip-lora.py
```
:::

::: {.cell .markdown}

The memory required is much smaller! Earlier, we saw that full fine-tuning went OOM on this model in this configuration, but now we can fit it easily. Although the base model weights are still loaded (they are used in the forward/backward pass), there is only a small number of trainable parameters (just the adapter weights), so the gradients are much smaller and the optimizer state is much smaller.

:::

::: {.cell .markdown}

We can also further reduce the memory required by quantizing the base model weights:

:::

::: {.cell .markdown}

Set `cfg` in `fine-tune-blip-lora.py` to this QLoRA configuration:

```python
cfg = {
    "model_name": "Salesforce/blip2-flan-t5-xxl",
    "lr": 5e-6,
    "batch_size": 64,
    "accumulate_grad_batches": 1,
    "precision": "32-true",
    "optim": "adamw",
    "act_ckpt": False,
    "max_epochs": 2,
    "max_steps": -1,
    "devices": 1,
    "strategy": "auto",
    "num_workers": 8,
    "limit_train_batches": 1.0,
    "enable_checkpointing": False,
    "num_train_samples": 512,
    "save_model": False,
    "use_lora": False,
    "use_qlora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
```

:::

::: {.cell .markdown}

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference). Also check the `[cfg]` output near the top of the run to confirm the script is using the config you intended.

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip-lora.py
```
:::


::: {.cell .markdown}

When you have finished, download this notebook - which includes the output of each experiment stage - from the Jupyter environment for later reference.

:::
