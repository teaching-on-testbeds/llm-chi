

# Large-scale model training on Chameleon

In this tutorial, we will practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

* gradient accumulation
* reduced precision
* parameter efficient fine tuning
* distributed training across multiple GPUs and with CPU offload

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC site.



## Experiment topology 

In this experiment, we will deploy a single bare metal instance with specific NVIDIA GPU capabilities.

* In the "Single GPU" section: To use `bfloat16` in our experiments with reduced precision, we need a GPU with [NVIDIA CUDA compute capability](https://developer.nvidia.com/cuda-gpus) 8.0 or higher. For example, some Chameleon nodes have A100 or A30 GPUs, which have compute capability 8.0. If we use a V100, with compute capability 7.0, the `bfloat16` capability is actually emulated in software.
* In the "Multiple GPU" section: To practice distributed training with multiple GPUs, will request a node with 4 GPUs. 


We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). 

For example, to find nodes with 4x GPUs: if we expand "Advanced Filters", check the "4" box under "GPU count", and then click "View", we can identify some suitable node types: `gpu_a100_pcie`, `gpu_a100_nvlink`, `gpu_v100`, or `gpu_v100_nvlink` at CHI@UC. (The [NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/)-type nodes have a high-speed interconnect between GPUs.)

(We will avoid `p100`-based node types for this experiment, because the P100 has less GPU RAM, and less compute capability.)



## Create a lease



To use bare metal resources on Chameleon, we must reserve them in advance. We can use the OpenStack graphical user interface, Horizon, to submit a lease for an A100 or V100 node at CHI@UC. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@UC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.



**If you plan to do "Single GPU" and "Multiple GPU" together in a 3-hour block**: 

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_a100_pcie` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the node type or name of the node you want to reserve.
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>llm_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
  * Click "Next".
On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node type that you identified earlier.
  * Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". Click on the lease to see an overview. It will show the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease. Make sure that the lease details are correct.



**If you plan to do "Single GPU" and "Multiple GPU" separately in two 2-hour blocks**: 

First, make a 2-hour reservation on a node with a single A100 80GB GPU. We will use a `compute_gigaio`, but avoid [`gigaio-compute-06`](https://chameleoncloud.org/hardware/node/sites/uc/clusters/chameleon/nodes/25ba0313-900d-4133-9bb3-9f622d743c2d/) which has no GPU.

* On the left side, click on "Reservations" > "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_gigaio` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC. You can use [WolframAlpha](https://www.wolframalpha.com/) or equivalent to convert to your local time zone.
* Once you have identified an available three-hour block in UTC time that works for you in your local time zone, make a note of:
  * the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
  * and the name of the node you want to reserve.
* Then, on the left side, click on "Reservations" > "Leases", and then click on "Create Lease":
  * set the "Name" to <code>llm_single_<b>netID</b></code> where in place of <code><b>netID</b></code> you substitute your actual net ID.
  * set the start date and time in UTC
  * modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
  * Click "Next".
On the "Hosts" tab, 
  * check the "Reserve hosts" box
  * leave the "Minimum number of hosts" and "Maximum number of hosts" at 1
  * in "Resource properties", specify the node name that you identified earlier.
  * Click "Next". Then, click "Create". (We won't include any network resources in this lease.)
  
Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease.

Next, make a 2-hour reservation on a node with 4x A100 or 4x V100 GPU. Repeat the steps above, but for an `gpu_a100_pcie` or `gpu_v100` node type, and use the lease name <code>llm_multi_<b>netID</b></code> (with your own net ID).



Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your GPU time.



At the beginning of your lease time, you will continue with the next step, in which you bring up a bare metal instance!




Before you begin, open this experiment on Trovi:

* Use this link: [Large-scale model training on Chameleon](https://chameleoncloud.org/experiment/share/39a536c6-6070-4ccf-9e91-bc47be9a94af) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it.

You will see several notebooks inside the `llm-chi` directory - look for the one titled `1_create_server.ipynb`. Open this notebook and continue there.



## Bring up a GPU server

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:


```python
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@UC")
```


Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:


```python
l = lease.get_lease(f"llm_netID") # or llm_single_netID, or llm_multi_netID
l.show()
```


The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting Run > Run Selected Cell and All Below from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!



We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. (Note that the reservation information is passed when we create the instance!) This will take up to 10 minutes.



```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-llm-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```


Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.



Then, we'll associate a floating IP with the instance, so that we can access it over SSH.


```python
s.associate_floating_ip()
```

```python
s.refresh()
s.check_connectivity()
```

```python
s.refresh()
s.show(type="widget")
```



## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.


```python
s.execute("git clone https://github.com/teaching-on-testbeds/llm-chi")
```



## Set up Docker with NVIDIA container toolkit

To use common deep learning frameworks like Tensorflow or PyTorch, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.


```python
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("docker run hello-world")
```


We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.


```python
# get NVIDIA container toolkit 
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
s.execute("sudo systemctl restart docker")
```



In the following cell, we will verify that we can see our NVIDIA GPUs from inside a container, by passing `--gpus-all`. (The `-rm` flag says to clean up the container and remove its filesystem when it finishes running.)



```python
s.execute("docker run --rm --gpus all ubuntu nvidia-smi")
```


Let's pull the actual container images that we are going to use, 

* For the "Single GPU" section: a Jupyter notebook server with PyTorch and CUDA libraries
* For the "Multiple GPU" section: a PyTorch image with NVIDIA developer tools, which we'll need in order to install DeepSpeed



## Pull and start container for "Single GPU" section

Let's pull the container:


```python
s.execute("docker pull quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1")
```



and get it running:


```python
s.execute("docker run -d -p 8888:8888 --gpus all --name torchnb quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1")
```



There's one more thing we must do before we can start out Jupyter server. Rather than expose the Jupyter server to the Internet, we are going to set up an SSH tunnel from our local terminal to our server, and access the service through that tunnel. 

Here's how it works: In your *local* terminal, run

```
ssh -L 8888:127.0.0.1:8888 -i ~/.ssh/id_rsa_chameleon cc@A.B.C.D
```

where, 

* instead of `~/.ssh/id_rsa_chameleon`, substitute the path to your key
* and instead of `A.B.C.D`, substitute the floating IP associated with your server

This will configure the SSH session so that when you connect to port 8888 locally, it will be forwarded over the SSH tunnel to port 8888 on the host at the other end of the SSH connection.

SSH tunneling is a convenient way to access services on a remote machine when you don't necessarily want to expose those services to the Internet (for example: if they are not secured from unauthorized access).



Finally, run


```python
s.execute("docker logs torchnb")
```


Look for the line of output in the form:

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

and copy it for use in the next section.



You will continue working from the next notebook! But, if you are planning to do the "Multiple GPU" section in the same lease, run the following cells too, to let the container image for the "Multiple GPU" section get pulled in the background. This image takes a loooooong time to pull, so it's important to get it started and leave it running while you are working in your other tab on the "Single GPU" section.




## Pull container for "Multiple GPU" section




```python
s.execute("docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
```




and let's also install some software on the host that we'll use in the "Multiple GPU" section:



```python
s.execute("sudo apt update; sudo apt -y install nvtop")
```





## Train a large model on a single GPU

In this section, we will practice strategies for training a large model on a single GPU. After completing this section, you should understand the effect of

* batch size
* gradient accumulation
* reduced precision/mixed precision
* parameter efficient fine tuning

on a large model training job.

This notebook will be executed inside a Jupyter interface **hosted on a GPU server instance on Chameleon**, NOT in the Chameleon Jupyter interface from which we launch experiments (provision servers, etc.) 




### Open the notebook on Colab

We should have already started a notebook server in a container on a Chameleon GPU host, and set up an SSH tunnel to this notebook server. Now, we will open this notebook in Google Colab and connect it to the runtime that you have in Chameleon. This is a convenient way to work, because the notebook and its outputs will be saved automatically in your Google Drive.

* Use this button to open the notebook in Colab: <a target="_blank" href="https://colab.research.google.com/github/teaching-on-testbeds/llm-chi/blob/main/workspace/2_single_gpu_a100.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
* Click "File > Save a Copy in Drive" to save it in your own Google Drive. Work in your copy, so that the outputs will be saved automatically.
* Next to the "Connect" button in the top right, there is a &#9660; symbol. Click on this symbol to expand the menu, and choose "Connect to a local runtime".
* Paste the `http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` you copied earlier into this space, and choose "Connect".

**Alternatively, if you prefer not to use Colab** (or can't, for some reason): just put the  `http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` URL you copied earlier into your browser to open the Jupyter interface directly. But, then you'll have to open a terminal in that Jupyter interface and run

```
wget https://raw.githubusercontent.com/teaching-on-testbeds/llm-chi/refs/heads/main/workspace/2_single_gpu_a100.ipynb
```

to get a copy of this notebook in that workspace.




Make sure that you can see the GPUs:


```python
!nvidia-smi
```



### Prepare LitGPT

For this tutorial, we will fine-tune an [TinyLlama](https://arxiv.org/abs/2401.02385) or [OpenLLaMA](https://github.com/openlm-research/open_llama) large language model using [`litgpt`](https://github.com/Lightning-AI/litgpt). LitGPT is a convenient wrapper around many PyTorch Lightning capabilities that makes it easy to fine-tune a GPU using a "recipe" defined in a YAML file. (We'll also try the Python API for LitGPT in the "Multiple GPU" section of this tutorial.)

Our focus will be exclusively on comparing the time and memory requirements of training jobs under different settings - we will completely ignore the loss of the fine-tuned model, and we will make some choices to reduce the overall time of our experiment (to fit in a short Chameleon lease) that wouldn't make sense if we really needed the fine-tuned model (e.g. using a very small fraction of the training data).

First, install LitGPT:


```python
!pip install 'litgpt[all]'==0.5.7 'lightning<2.5.0.post0'
```




then, download the foundation models:



```python
!litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```


```python
!litgpt download openlm-research/open_llama_3b
```

```python
!litgpt download openlm-research/open_llama_7b
```


```python
!litgpt download openlm-research/open_llama_13b
```





Also, get the "recipes" that we will use for LLM fine-tuning. Using the file browser on the left side, look at the contents of the "config" directory.



```python
!git clone https://github.com/teaching-on-testbeds/llm-chi/
!mv llm-chi/workspace/config .
```




### Experiment: Baseline

As a baseline, let's try an epoch of fine-tuning the TinyLlama-1.1B, using full precision and a batch size of 32:



```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 32
```


This will fail because the training job won't fit in our 80GB GPU memory.




### Experiment: Reduced batch size

But with a smaller batch size, it fits easily:



```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 8 --train.micro_batch_size 8
```


Make a note of the training time and memory, which is printed at the end of the training job.



### Experiment: Gradient accumulation

By using gradient accumulation to "step" only after a few "micro batches", we can train with a larger effective "global" batch size, with minimal effect on the memory required:



```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8
```


Make a note of the training time and memory, which is printed at the end of the training job.



### Experiment: Reduced precision

With a "brain float16" format for numbers, instead of "float32", we can further reduce the memory required, although this representation is less precise:



```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```


Make a note of the training time and memory, which is printed at the end of the training job.







### Experiment: Mixed precision

With mixed precision, we get back some of the lost precision in the results, at the cost of some additional memory and time:



```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-mixed
```


Make a note of the training time and memory, which is printed at the end of the training job.



### Experiment: Larger model - 3b

We've gained so much GPU memory back with these techniques, we can even train a larger model. Let's switch from the 1.1B to the 3B model:


```python
!litgpt finetune_full --config config/open-llama-3b-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```


Make a note of the training time and memory, which is printed at the end of the training job.



### Experiment: Larger model - 7b

If we reduce the batch size again, we can even train a 7b model:



```python
!litgpt finetune_full --config config/open-llama-7b-full.yaml --train.global_batch_size 16 --train.micro_batch_size 4 --precision bf16-true
```


Make a note of the training time and memory, which is printed at the end of the training job.




### Experiment: Larger model - 13b

Even with the smallest possible batch size, we can't train a 13B model:



```python
!litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true
```


this will fail with an "out of memory" error. But, if we switch from the Adam optimizer (which has two state values per parameter) to SGD, we can train a 13B model. It's *verrrrry* slow, though, so we won't even train it for a full epoch - just 25 "steps", so we can get an idea of the memory required:


```python
!litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true --optimizer SGD --train.max_steps 25
```




### Experiment: Parameter efficient fine tuning

If we are only fine-tuning, not training a model from scratch, we can also consider LoRA and QLoRA. Let's try it first with our 1.1B model:



```python
!litgpt finetune --config config/tiny-llama-lora.yaml
```


The memory required is *shockingly* small! We can see it with our 3B and 7B models, too:

```python
!litgpt finetune --config config/open-llama-3b-lora.yaml
```

```python
!litgpt finetune --config config/open-llama-7b-lora.yaml
```


We can also further reduce the memory required with quantization:



```python
!litgpt finetune --config config/open-llama-7b-lora.yaml --quantize bnb.nf4
```




Even the 13B model can be trained quickly with minimal memory required, using LoRA:


```python
!litgpt finetune --config config/open-llama-13b-lora.yaml
```




## Train a large model on multiple GPUs - 4x A100 80GB

In this section, we will practice strategies for training a large model using distributed processes across multiple GPUs. This section requires a host with 4x A100 80GB GPUs.

After completing this section, you should understand the effect of

* distributed data parallelism
* and fully sharded data parallelism

on a large model training job.

You will execute the commands in this section either inside an SSH session on the Chameleon "node-llm" server, or inside a container that runs on this server. You will need **two** terminals arranged side-by-side or vertically, and in both terminals, use SSH to connect to the "node-llm" server.



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



### Start `nvtop` on the host

In your second terminal session, start `nvtop`, which we will use to monitor the resource usage of the NVIDIA GPUs on the host:

```bash
# run on node-llm
nvtop
```

and leave it running throughout all the experiments in this section.



### Experiment: OpenLLaMA 7b model on a single A100 80GB


We previously noted that we can train an OpenLLaMA 7b model on a single A100 80GB GPU with bf16 precision and batch size 4, and that this setting would essentially max out the available GPU memory on the A100 80GB.

Inside the container, we'll repeat this test (using the Python API for `litgpt` instead of its command line interface) (and, we won't use gradient accumulation this time):

```bash
# run inside pytorch container
python3 a100_llama7b_1device.py
```

As it runs, note in `nvtop` that only one GPU is used. We will see that for GPU 0, the GPU utilization is close to 100% and the GPU memory utilization is also high, but the other GPUs have zero utilization. Also note that in the list of processes, there is a single process running on device 0.

Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.


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



### Experiment: OpenLLaMA 7b model on 4x A100 80GB with DDP


Now, we'll repeat the same experiment with DDP across 4 GPUs! Inside the container, run

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


<!--

Note to self:
a100_llama7b_4ddp.py

Allocated: 70.44 GB, Reserved: 78.06 GB
Allocated: 69.62 GB, Reserved: 78.12 GB
Allocated: 73.98 GB, Reserved: 78.30 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:43<00:00,  0.61it/s, v_num=6, train_loss=1.560]


-->



### Experiment: OpenLLaMA 7b model on 4x A100 80GB with FSDP


With DDP, we have a larger effective batch size (since 4 GPUs process a batch in parallel), but no memory savings. With FSDP, we can shard optimizer state, gradients, and parameters across GPUs, to also reduce the memory required.

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


<!--

Note to self:
a100_llama7b_4fsdp.py (FULL_SHARD)

Allocated: 31.96 GB, Reserved: 56.00 GB
Allocated: 32.79 GB, Reserved: 55.98 GB
Allocated: 36.33 GB, Reserved: 55.76 GB
Epoch 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:53<00:00,  0.57it/s, v_num=9, train_loss=1.430]`Trainer.fit` stopped: `max_epochs=1` reached.


-->


### Experiment: OpenLLaMA 7b model on 4x A100 80GB with FSDP and larger batch size


Because of the memory savings achieved by FSDP, we can increase the batch size (and potentially achieve faster training times) without running out of memory. 

Inside the container, run:


```bash
# run inside pytorch container
python3 a100_llama7b_4fsdp_8batch.py
```

In this training script, we've changed the `batch_size` to 8.

As it runs, note in `nvtop` that the GPUs again have high memory utilization. Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the `python3` command finishes running in the container, note the training time (displayed to the right of the progress bar) and the memory usage reported in the output, and take a screenshot for later reference.


<!--

Note to self:
batch size 8:

Allocated: 62.95 GB, Reserved: 67.67 GB
Allocated: 59.51 GB, Reserved: 62.71 GB
Allocated: 62.95 GB, Reserved: 64.74 GB
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:56<00:00,  0.43it/s, v_num=13, train_loss=1.610]`Trainer.fit` stopped: `max_epochs=1` reached.


a100_llama7b_4fsdp_12batch.py (FULL_SHARD)



-->


### (Optional) Experiment: OpenLLaMA 13b model on 4x A100 80GB with CPU optimizer offload via DeepSpeed

Finally, as an optional experiment, we can try training a much bigger model - the 13B OpenLLaMA model - using a combination of:

* sharding parameters and gradients across GPUs, as before
* and offloading the optimizer state to CPU


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





#### Debugging note

Note: if any training job crashes due to OOM, you can ensure all of the distributed processes are stopped by running

```bash
# run inside pytorch container
pkill -9 python
```




<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>