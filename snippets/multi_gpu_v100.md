

::: {.cell .markdown}

## Train a large model on multiple GPUs - 4x V100 32GB

In this section, we will practice strategies for training a large model using distributed processes across multiple GPUs. This section requires a host with 4x V100 GPUs with 32GB video RAM.

> **Note**: If you have already done the "Multiple GPU" section on a 4x A100 GPU instance, you will skip this section! This is just an alternative version of the same ideas, executed on different hardware.

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
litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
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

### Experiment: TinyLlama 1.1B model on a single V100 32GB


We previously noted that we can train an TinyLlama 1.1B model on a single GPU with bf16 precision and micro batch size 8, with less than 32GB of RAM (to fit in the memory of our V100).

Now, we'll repeat this test using the Python API for `litgpt` instead of its command line interface. You may view [v100_llama1b_1device.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/v100_llama1b_1device.py) in our Github repository. Run it inside the container with:

```bash
# run inside pytorch container
python3 v100_llama1b_1device.py
```

As it runs, note in `nvtop` that only one GPU is used. We will see that for GPU 0, the GPU utilization is close to 100% and the GPU memory utilization is also high, but the other GPUs have zero utilization. Also note that in the list of processes, there is a single process running on device 0.

Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

:::

<!--

Note to self:
v100_llama1b_1device.py

Allocated: 23.92 GB, Reserved: 25.16 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [05:08<00:00,  0.41it/s, v_num=6, train_loss=1.120]

-->


::: {.cell .markdown}

### Experiment: TinyLlama 1.1B model on 4x V100 32GB with DDP


Now, we'll repeat the same experiment with DDP across 4 GPUs!  You may view [v100_llama1b_4ddp.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/v100_llama1b_4ddp.py) in our Github repository. Inside the container, run

```bash
# run inside pytorch container
python3 v100_llama1b_4ddp.py
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
v100_llama1b_4ddp.py

Allocated: 23.75 GB, Reserved: 31.22 GB
Allocated: 23.10 GB, Reserved: 27.19 GB
Allocated: 25.27 GB, Reserved: 28.74 GB
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [01:36<00:00,  0.33it/s, v_num=7, train_loss=2.050]


-->


::: {.cell .markdown}

### Experiment: TinyLlama 1.1B model on 4x V100 32GB with FSDP


With DDP, we have a larger effective batch size (since 4 GPUs process a batch in parallel), but no memory savings. With FSDP, we can shard optimizer state, gradients, and parameters across GPUs, to also reduce the memory required.

You may view [v100_llama1b_4fsdp.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/v100_llama1b_4fsdp.py) in our Github repository. 

Inside the container, run:


```bash
# run inside pytorch container
python3 v100_llama1b_4fsdp.py
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
v100_llama1b_4fsdp.py (FULL_SHARD)

Allocated: 19.13 GB, Reserved: 26.12 GB
Allocated: 20.65 GB, Reserved: 25.27 GB
Allocated: 21.35 GB, Reserved: 24.99 GB
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [02:00<00:00,  0.27it/s, v_num=8, train_loss=2.030]


-->

::: {.cell .markdown}

### Experiment: OpenLLaMA 3B model on 4x V100 32GB with FSDP


Because of the memory savings achieved by FSDP, we can train a larger model without running out of memory. 

You may view [v100_llama3b_4fsdp.py](https://github.com/teaching-on-testbeds/llm-chi/blob/main/torch/v100_llama3b_4fsdp.py) in our Github repository. 


Inside the container, run:


```bash
# run inside pytorch container
python3 v100_llama3b_4fsdp.py
```

In this training script, we've changed the model to `openlm-research/open_llama_3b` and the batch size to `1`, and turned off gradient accumulation. Also, since this is slow, we're training on a smaller fraction of the data than in our previous experiments.

Take a screenshot of the `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.

Note that we cannot train this model on a single V100 without running out of memory - try


```bash
# run inside pytorch container
python3 v100_llama3b_1device.py
```

which runs on one GPU but otherwise has the same configuration, and observe that you get an OOM error.


:::

<!--

Note to self:

Allocated: 13.12 GB, Reserved: 28.35 GB
Allocated: 13.26 GB, Reserved: 27.56 GB
Allocated: 13.27 GB, Reserved: 27.54 GB

Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:04<00:00,  0.16it/s, v_num=16, train_loss=1.880]


-->


::: {.cell .markdown}

#### Debugging note

Note: if any training job crashes due to OOM, you can ensure all of the distributed processes are stopped by running

```bash
# run inside pytorch container
pkill -9 python
```
:::


