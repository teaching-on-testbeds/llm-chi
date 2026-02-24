::: {.cell .markdown}

## Train a large model on multiple GPUs

In this section, we will practice strategies for training a large model using distributed processes across multiple GPUs. This section requires a host with 4x GPUs.

After completing this section, we should understand the effect of

* distributed data parallelism
* and learning-rate scaling with larger world size

on a large model training job.

:::

::: {.cell .markdown}

Make sure that we can see the GPUs inside the container:

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-multi
nvidia-smi
```
:::

::: {.cell .markdown}

Throughout these experiments, we will monitor GPU utilization with `nvtop`. Open a terminal in Jupyter (File > New > Terminal), then run:

```bash
# runs in the Jupyter service on node-llm-multi
nvtop
```

Keep this running while training.

:::

::: {.cell .markdown}

Before running the training script, download and unpack the dataset snapshot that we will use in this lab.

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-multi
cd ~/work
mkdir -p data
wget -O data/gourmetgram_caption.tar.gz "https://nyu.box.com/shared/static/g3qw3g5j7l8dkvyf02a9afuuo9grs3g3.gz"
mkdir -p data/gourmetgram_caption
tar -xzf data/gourmetgram_caption.tar.gz -C data/gourmetgram_caption --strip-components=1
```
:::

::: {.cell .markdown}

The training script now reads the dataset from `./data/gourmetgram_caption`.

:::


::: {.cell .markdown}

### Experiment 1: Single-GPU baseline on the multi-GPU node

We will start with a baseline for single-GPU performance before turning on distributed training.

:::



::: {.cell .markdown}

Set `cfg` in `fine-tune-blip.py` to:

```python
cfg = {
    "model_name": "Salesforce/blip2-opt-2.7b",
    "lr": 2e-5,
    "batch_size": 16,
    "accumulate_grad_batches": 4,
    "precision": "bf16-true",
    "optim": "adamw",
    "act_ckpt": False,
    "max_epochs": 4,
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

Then, run:

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

As it runs, note in `nvtop` that only one GPU is used. For GPU 0, GPU utilization is high and memory utilization is also high, while the other GPUs have zero utilization.

Also note that in the list of processes, there is a single process running on device 0.

Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the training run finishes, note the training time and the memory summary printed by the script.

:::

::: {.cell .markdown}

### Experiment 2: DDP with same batch settings and scaled LR

Now, we will repeat the same experiment with DDP across 4 GPUs, while keeping the same per-device batch settings.

With DDP, each GPU processes its own batch, so effective global batch is 4x larger than Experiment 1.

We scale the learning rate by 4x to keep the update magnitude more comparable.

DDP also allocates gradient communication buckets that are about the same order as total parameter size. Even with `gradient_as_bucket_view=True`, we still see this extra bucket memory, so we use the smaller model here.

:::

::: {.cell .markdown}

In `cfg`, change:

* `"devices": 1` -> `"devices": 4`
* `"strategy": "auto"` -> `"strategy": "ddp"`
* `"lr": 2e-5` -> `"lr": 8e-5`

Leave all other values the same. Then, run:

:::

::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

Note that it may take a minute or two for the training job to start.

As it runs, note in `nvtop` that four GPUs are used, all with high utilization, and that four processes are listed. Take a screenshot of this `nvtop` display while the script is running, for later reference.

Check the memory logs printed by the Python script. Note that in distributed training, the `Other` value is a residual (`allocated - params - grads - optim`), that includes non-activation allocations such as communication buffers and collective buckets.

When the training run finishes, note the training time and memory summary printed by the script.

Compare this run with Experiment 1:

* per-GPU memory will be larger than in single-GPU. Per-device batch settings are unchanged, but we also have memory overhead for communication buffers and collective buckets.
* total throughput may or may not improve, because now four GPUs are working, but we also have communication overhead

:::

::: {.cell .markdown}

### Experiment 3: FSDP

Now we will switch from DDP to FSDP, while keeping the same batch settings and learning rate as Experiment 2.

FSDP shards model states across GPUs, so it can reduce per-GPU memory pressure.

:::

::: {.cell .markdown}

In our script, we define a set of layer classes and let Lightning FSDP auto-wrap those layers. Here, "wrap" means replacing each matching layer module with an FSDP-managed version of that module, so its parameters, gradients, and optimizer states can be sharded across GPUs instead of kept as full copies on every GPU.

```python
_blip2_layer_cls = {Blip2EncoderLayer, Blip2QFormerLayer, OPTDecoderLayer}
```

and then the strategy passed to the Lightning `Trainer` is:

```python
FSDPStrategy(auto_wrap_policy=_blip2_layer_cls)
```

:::

::: {.cell .markdown}

In `cfg`, change:

* `"strategy": "ddp"` -> `"strategy": "fsdp"`

Leave all other values the same as Experiment 2.

Then, run:

:::


::: {.cell .code}
```bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```
:::

::: {.cell .markdown}

As it runs, note in `nvtop` that four GPUs are used, and pay attention to memory usage on each GPU. Compare this run with Experiment 2.

When the training run finishes, note the training time and memory summary printed by the script.


:::
