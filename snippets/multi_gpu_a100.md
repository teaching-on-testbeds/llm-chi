

::: {.cell .markdown}

## Train a large model on multiple GPUs - 4x A100 80GB

In this section, we will practice strategies for training a large model using distributed processes across multiple GPUs. This section requires a host with 4x A100 80GB GPUs.

> **Note**: If you have reserved a 4x V100 GPU instance, skip to the V100 section!


After completing this section, you should understand the effect of

* distributed data parallelism
* and fully sharded data parallelism

on a large model training job.

You may view the Python code we will execute in this experiment [in our Github repository](https://github.com/teaching-on-testbeds/llm-chi/tree/main/torch).


You will execute the commands in this section either inside an SSH session on the Chameleon "node-llm" server, or inside a container that runs on this server. You will need **two** terminals arranged side-by-side or vertically, and in both terminals, use SSH to connect to the "node-llm" server.

:::

::: {.cell .markdown}

### Start the container

We will run code inside a container that has:

* PyTorch
* NVIDIA CUDA and NVIDIA CUDA developer tools, because these will be needed to install DeepSpeed

First, make sure there are no other containers running, because we will need exclusive access to the GPUs:

```bash
# run on node-llm
docker ps
```

If any containers are still running, stop them with

```bash
# run on node-llm
docker stop CONTAINER
```

(substituting the container name or ID in place of `CONTAINER`.)

Then, start the PyTorch + NVIDIA CUDA and NVIDIA CUDA developer tools container with

```bash
# run on node-llm
docker run -it -v /home/cc/llm-chi/torch:/workspace --gpus all --ipc host pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

Note that:

* `-v /home/cc/llm-chi/torch:/workspace` sets up a bind mount, so the contents of the `/home/cc/llm-chi/torch` directory on the "node-llm" host (which has the code we'll use in this section!) will appear in the `/workspace` directory of the container.
* `--gpus all` passes through all of the host's GPUs to the container
* `--ipc host` says to use the host namespace for inter-process communication, which will improve performance. (A slightly more secure alternative would be to set `--shm-size` to a large value, to increase the memory available for inter-process communication, but for our purposes `--ipc host` is fine and more convenient.)

:::

::: {.cell .markdown}

### Install software in the container

Inside the container, install a few Python libraries:

```bash
# run inside pytorch container
pip install 'litgpt[all]'==0.5.7 'lightning<2.5.0.post0'
```

and download the foundation model we are going to fine-tune:


```bash
# run inside pytorch container
litgpt download openlm-research/open_llama_7b
```

:::

::: {.cell .markdown}

### Start `nvtop` on the host

In your second terminal session, start `nvtop`, which we will use to monitor the resource usage of the NVIDIA GPUs on the host:

```bash
# run on node-llm
nvtop
```

and leave it running throughout all the experiments in this section.

:::

::: {.cell .markdown}

### Experiment: OpenLLaMA 7b model on a single A100 80GB


We previously noted that we can train an OpenLLaMA 7b model on a single A100 80GB GPU with bf16 precision and batch size 4, and that this setting would essentially max out the available GPU memory on the A100 80GB.

Now, we'll repeat this test using the Python API for `litgpt` instead of its command line interface (and, we won't use gradient accumulation this time). You may view [a100_llama7b_1device.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/a100_llama7b_1device.py) in our Github repository. Run it inside the container with:

```bash
# run inside pytorch container
python3 a100_llama7b_1device.py
```

As it runs, note in `nvtop` that only one GPU is used. We will see that for GPU 0, the GPU utilization is close to 100% and the GPU memory utilization is also high, but the other GPUs have zero utilization. Also note that in the list of processes, there is a single process running on device 0.

Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

:::

<!--

Note to self:
a100_llama7b_1device.py

with gradient accumulation 4:

Allocated: 61.99 GB, Reserved: 73.30 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 400/400 [02:45<00:00,  2.41it/s, v_num=3, train_loss=1.590]`Trainer.fit` stopped: `max_epochs=1` reached.

Without gradient accumulation:

Allocated: 61.97 GB, Reserved: 71.35 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 400/400 [03:32<00:00,  1.89it/s, v_num=7, train_loss=2.120]

-->


::: {.cell .markdown}

### Experiment: OpenLLaMA 7b model on 4x A100 80GB with DDP


Now, we'll repeat the same experiment with DDP across 4 GPUs!  You may view [a100_llama7b_4ddp.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/a100_llama7b_4ddp.py) in our Github repository. Inside the container, run

```bash
# run inside pytorch container
python3 a100_llama7b_4ddp.py
```

In this training script, we've exchanged

```python
    devices=1,
```

for 

```python
    devices=4,
    strategy=DDPStrategy(),
```

Note that it may take a minute or two for the training job to start.

As it runs, note in `nvtop` that four GPUs are used, all with high utilization, and that four processes are listed. Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

:::

<!--

Note to self:
a100_llama7b_4ddp.py

Allocated: 70.44 GB, Reserved: 78.06 GB
Allocated: 69.62 GB, Reserved: 78.12 GB
Allocated: 73.98 GB, Reserved: 78.30 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:43<00:00,  0.61it/s, v_num=6, train_loss=1.560]


-->


::: {.cell .markdown}

### Experiment: OpenLLaMA 7b model on 4x A100 80GB with FSDP


With DDP, we have a larger effective batch size (since 4 GPUs process a batch in parallel), but no memory savings. With FSDP, we can shard optimizer state, gradients, and parameters across GPUs, to also reduce the memory required.

You may view [a100_llama7b_4fsdp.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/a100_llama7b_4fsdp.py) in our Github repository. 

Inside the container, run:


```bash
# run inside pytorch container
python3 a100_llama7b_4fsdp.py
```

In this training script, we've exchanged

```python
    strategy=DDPStrategy(),
```

for 

```python
    strategy=FSDPStrategy(sharding_strategy='FULL_SHARD'),
```

As it runs, note in `nvtop` that four GPUs are used, with high utilization of the GPU but lower utilization of its memory. Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

:::

<!--

Note to self:
a100_llama7b_4fsdp.py (FULL_SHARD)

Allocated: 31.96 GB, Reserved: 56.00 GB
Allocated: 32.79 GB, Reserved: 55.98 GB
Allocated: 36.33 GB, Reserved: 55.76 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:53<00:00,  0.57it/s, v_num=9, train_loss=1.430]`Trainer.fit` stopped: `max_epochs=1` reached.


-->

::: {.cell .markdown}

### Experiment: OpenLLaMA 7b model on 4x A100 80GB with FSDP and larger batch size


Because of the memory savings achieved by FSDP, we can increase the batch size (and potentially achieve faster training times) without running out of memory. 

You may view [a100_llama7b_4fsdp_8batch.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/a100_llama7b_4fsdp_8batch.py) in our Github repository. 


Inside the container, run:


```bash
# run inside pytorch container
python3 a100_llama7b_4fsdp_8batch.py
```

In this training script, we've changed the `batch_size` to 8.

As it runs, note in `nvtop` that the GPUs again have high memory utilization. Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

:::

<!--

Note to self:
batch size 8:

Allocated: 62.95 GB, Reserved: 67.67 GB
Allocated: 59.51 GB, Reserved: 62.71 GB
Allocated: 62.95 GB, Reserved: 64.74 GB
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:56<00:00,  0.43it/s, v_num=13, train_loss=1.610]`Trainer.fit` stopped: `max_epochs=1` reached.


a100_llama7b_4fsdp_12batch.py (FULL_SHARD)



-->

::: {.cell .markdown}

### (Optional) Experiment: OpenLLaMA 13b model on 4x A100 80GB with CPU optimizer offload via DeepSpeed

Finally, as an optional experiment, we can try training a much bigger model - the 13B OpenLLaMA model - using a combination of:

* sharding parameters and gradients across GPUs, as before
* and offloading the optimizer state to CPU

You may view [a100_llama13b_deepspeed.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/a100_llama13b_deepspeed.py) in our Github repository. 


For this experiment, we'll install `deepspeed`:

```bash
# run inside pytorch container
DS_BUILD_CPU_ADAM=1 pip install deepspeed
```

and download the 13b model:

```bash
# run inside pytorch container
litgpt download openlm-research/open_llama_13b
```

Now, we can run

```bash
# run inside pytorch container
python3 a100_llama13b_deepspeed.py
```

In this training script, besides for replacing the 7B model with the 13B model:

* We swapped out our previous PyTorch Adam optimizer for `DeepSpeedCPUAdam`
* We changed the training strategy from FSDP to

```
    strategy=DeepSpeedStrategy(
        stage=3,                 # Similar to FULL_SHARD
        offload_optimizer=True   # Enable CPU offloading of optimizer
    ),
```

As it runs, note in `nvtop` that especially near the end of the step, the GPUs will be underutilized as they wait for CPU. 

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.


:::


::: {.cell .markdown}

#### Debugging note

Note: if any training job crashes due to OOM, you can ensure all of the distributed processes are stopped by running

```bash
# run inside pytorch container
pkill -9 python
```
:::


