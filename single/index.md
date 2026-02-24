# Large-scale model training on Chameleon - single GPU

In this tutorial, we will practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

-   gradient accumulation
-   reduced precision
-   parameter efficient fine tuning

To run this experiment, you should have already created an account on Chameleon, and become part of a project.

You must also have added your SSH key to the CHI@UC site (to use an A100 GPU) or KVM@TACC site (to use an H100 GPU).

## Experiment topology

In this experiment, we will deploy a single instance with a GPU. We have "tuned" this experiment for two specific GPU types:

-   an A100 with 80 GB VRAM (available on most `compute_gigaio` bare metal instances at CHI@UC)
-   or an H100 with 94GB VRAM (available in the `g1.h100.pci.1` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

You are currently viewing the A100 version of the instructions, but H100 instructions are also available at [index_h100](index_h100).

## Create a lease

To use a GPU instance on Chameleon, we must reserve it in advance. GPU instances are much more in-demand than other resource types, and so we typically cannot make a reservation "on the spot" to use one.

We can use the OpenStack graphical user interface, Horizon, to reserve a GPU in advance. To access this interface,

-   from the [Chameleon website](https://chameleoncloud.org/hardware/)
-   click "Experiment" \> "CHI@UC"
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

Reserve a 2 hr 50 minute block on a node with a single A100 80GB GPU. We will use `compute_gigaio`.

-   On the left side, click on "Reservations" \> "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_gigaio` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC, so you will need to convert to your local time zone.
-   Once you have identified an available 2 hr 50 minute block in UTC time that works for you in your local time zone, make a note of:
    -   the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
    -   and the name of the node you want to reserve.
-   Then, on the left side, click on the name of the node you want to reserve:
    -   set the "Name" to `llm_single_netID`, replacing `netID` with your actual net ID.
    -   set the start date and time in UTC
    -   modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
    -   Click "Next".
-   On the "Hosts" tab, confirm that the node you selected is listed in the "Resource properties" section, and click "Next".
-   Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease.

At the beginning of your lease time, continue with `2_create_server.ipynb`.

Before you begin, open this experiment on Trovi:

-   Use this link: [Large-scale model training on Chameleon](https://chameleoncloud.org/experiment/share/39a536c6-6070-4ccf-9e91-bc47be9a94af) on Trovi
-   Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it.

Inside the `llm-chi` directory, open the `single` subdirectory. You will see several notebooks - look for the one titled `2_create_server.ipynb`. Open this notebook and continue there.

## Bring up a GPU server

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

``` python
# run in Chameleon Jupyter environment
from chi import server, context, lease
import os

context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@UC")
```

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

``` python
# run in Chameleon Jupyter environment
l = lease.get_lease(f"llm_single_netID")
l.show()
```

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook sets up the instance and the experiment environment, which cumulatively takes a while - bringing up the instance can take some time, and building the container image can take some time.

But, you can be mostly hands-off in this stage. You can save time by clicking on this cell, then selecting Run \> Run Selected Cell and All Below from the Jupyter menu.

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image.

Bare metal instances can take much longer than VM instances to bring up, and the `gigaio` nodes in particular take even longer - up to 30 minutes.

So if it takes a while to build the instance, you just need to be patient - as long as it does not show the instance in `ERROR` state, it's working as expected.

``` python
# run in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this suffix
s = server.Server(
    f"node-llm-single-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

``` python
# run in Chameleon Jupyter environment
s.associate_floating_ip()
```

``` python
# run in Chameleon Jupyter environment
s.refresh()
s.check_connectivity()
```

In the output below, make a note of the floating IP that has been assigned to your instance.

``` python
# run in Chameleon Jupyter environment
s.refresh()
s.show(type="widget")
```

## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.

``` python
# run in Chameleon Jupyter environment
s.execute("git clone https://github.com/teaching-on-testbeds/llm-chi")
```

## Set up Docker with NVIDIA container toolkit

To use common deep learning frameworks like Tensorflow or PyTorch, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

``` python
# run in Chameleon Jupyter environment
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("docker run hello-world")
```

We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.

``` python
# run in Chameleon Jupyter environment
# get NVIDIA container toolkit 
s.execute("curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list")
s.execute("sudo apt update")
s.execute("sudo apt-get install -y nvidia-container-toolkit")
s.execute("sudo nvidia-ctk runtime configure --runtime=docker")
# for https://github.com/NVIDIA/nvidia-container-toolkit/issues/48
s.execute("sudo jq 'if has(\"exec-opts\") then . else . + {\"exec-opts\": [\"native.cgroupdriver=cgroupfs\"]} end' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp > /dev/null && sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json")
s.execute("sudo systemctl restart docker")
```

In the following cell, we will verify that we can see our NVIDIA GPUs from inside a container, by passing `--gpus all`. (The `-rm` flag says to clean up the container and remove its filesystem when it finishes running.)

``` python
# run in Chameleon Jupyter environment
s.execute("docker run --rm --gpus all ubuntu nvidia-smi")
```


## Build and start container for "Single GPU" section

Let's build the container image that we are going to use for this lab from `single/docker/Dockerfile`. It may take 10-15 minutes to build this container image.

You may view this Dockerfile in our Github repository: [single/docker/Dockerfile](https://github.com/teaching-on-testbeds/llm-chi/blob/main/single/docker/Dockerfile).

This image starts from the Jupyter Pytorch CUDA12 stack and adds the pieces we need for this lab:

-   CUDA toolkit 12.8 so we can build/install packages that require CUDA tooling
-   `nvtop` for NVIDIA GPU monitoring
-   ML Python libraries: notably, Lightning, Transformers and related libraries, and BitsAndBytes
-   DeepSpeed for CPU offload experiments with larger models

``` python
# run in Chameleon Jupyter environment
s.execute("docker build -t llm-jupyter:latest ~/llm-chi/single/docker")
```

and get it running:

``` python
# run in Chameleon Jupyter environment
s.execute("docker run --rm -d -p 8888:8888 -v /home/cc/llm-chi/single/workspace:/home/jovyan/work --gpus all --name jupyter llm-jupyter:latest")
```

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container on the `node-llm-<username>` instance:

``` python
# run in Chameleon Jupyter environment
s.execute("docker exec jupyter jupyter server list")
```

Look for a line like

    http://localhost:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface.

Continue with the notebook located inside that workspace.

## Train a large model on a single GPU

In this section, we will practice strategies for training a large model on a single GPU. After completing this section, you should understand the effect of

-   batch size
-   gradient accumulation
-   reduced precision/mixed precision
-   CPU offload
-   activation checkpointing
-   parameter efficient fine tuning

on a large model training job.

Make sure that you can see the GPU inside the container:

``` bash
# runs in the Jupyter service on node-llm-single
nvidia-smi
```

Throughout these experiments, we will monitor GPU compute and memory utilization with `nvtop`. Open a terminal in the Jupyter service on node-llm-single (File \> New \> Terminal) and in it, run

``` bash
# runs in the Jupyter service on node-llm-single
nvtop
```

In this display,

-   `GPU%` tells us how busy the GPU compute cores are. Low values with long training time often mean the GPU is waiting (for data, CPU work, or synchronization).
-   `GPU mem%` tells us how much GPU VRAM is currently in use. If this approaches 100%, we are close to OOM.

We will refer back to this display throughout the experiment.

Before running the training scripts, download and unpack the dataset snapshot that we will use in this lab.

``` bash
# runs in the Jupyter service on node-llm-single
cd ~/work
mkdir -p data
wget -O data/gourmetgram_caption.tar.gz "https://nyu.box.com/shared/static/g3qw3g5j7l8dkvyf02a9afuuo9grs3g3.gz"
mkdir -p data/gourmetgram_caption
tar -xzf data/gourmetgram_caption.tar.gz -C data/gourmetgram_caption --strip-components=1

ls -lah data/gourmetgram_caption
```

The training scripts now read the dataset from `./data/gourmetgram_caption`.

### PyTorch Lightning workflow

In this section, we will run a BLIP-2 fine-tuning script (`fine-tune-blip.py`) that is built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

PyTorch Lightning helps us keep the training loop logic mostly fixed while we change strategy using config values like:

-   batch size
-   gradient accumulation
-   precision mode (`32-true`, `bf16-mixed`, `bf16-true`)
-   activation checkpointing
-   optimizer choice
-   LoRA/QLoRA options

This means we can compare memory and training time across many settings by editing one config dictionary in a script, instead of rewriting training code each time.

Our focus will be on comparing time and memory requirements under different settings - we aren't trying to optimize for model quality.

First, let's briefly review the training script, which is written for Pytorch Lightning. Lightning is Pytorch with less boilerplate and some additional functionality baked in, including things like distributed training.

A basic Lightning module for our image captioning model would look something like this:

``` python
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

``` python
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

``` python
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

-   `_phase0_before_fwd`: right before the forward pass
-   `_phase1_after_fwd`: after forward, before backward
-   `_phase2_after_bwd`: after backward
-   `_phase3_after_opt_step`: after optimizer step (when optimizer state is materialized)

Lightning makes it easy for us to define custom code that runs at various points in the forward pass, backward pass, or optimizer step.

Each memory snapshot will include:

-   `Params`: model weights in memory
-   `Grads`: gradient tensors
-   `Optim`: optimizer state tensors
-   `Acts`: activation memory (Note: we actually don't have a good way to estimate this exactly, so we are assuming "whatever is left" is activation memory usage. In fact, this number as reported also includes other small non-activation items stored in memory, which are cumulatively less than 1GB.)
-   `Allocated/peak`: current and max allocated CUDA memory seen so far
-   `Reserved`: memory held by the CUDA allocator cache (can stay high even after tensors are freed)

and it will print these in step 0, step 1, and then every 50 steps after that.

Open `fine-tune-blip.py` in the Jupyter file browser. We will use a baseline config first, then edit only a few keys at each experiment stage.

For every experiment in this notebook, we will follow the same run loop:

1.  edit the requested keys in `cfg`
2.  run `python fine-tune-blip.py` in a terminal cell
3.  record whether it succeeds or fails, plus run time and memory output

For quick reference, this table summarizes the configuration used in each full fine-tuning experiment.

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Experiment                 Model                 bs/acc   precision      optim             act_ckpt   strategy                      max_steps   Notes
  -------------------------- --------------------- -------- -------------- ----------------- ---------- ----------------------------- ----------- --------------------------------------------------
  Baseline                   `blip2-opt-2.7b`      `32/1`   `32-true`      `adamw`           `False`    `auto`                        `-1`        `lr=5e-6`, `num_train_samples=512`, expected OOM

  Reduced batch size         `blip2-opt-2.7b`      `16/1`   `32-true`      `adamw`           `False`    `auto`                        `-1`        

  Gradient accumulation      `blip2-opt-2.7b`      `16/4`   `32-true`      `adamw`           `False`    `auto`                        `-1`        

  Grad accum + LR x4 rerun   `blip2-opt-2.7b`      `16/4`   `32-true`      `adamw`           `False`    `auto`                        `-1`        `lr=2e-5`

  Reduced precision          `blip2-opt-2.7b`      `16/4`   `bf16-true`    `adamw`           `False`    `auto`                        `-1`        

  Mixed precision            `blip2-opt-2.7b`      `16/4`   `bf16-mixed`   `adamw`           `False`    `auto`                        `-1`        

  Larger model               `blip2-opt-6.7b`      `16/4`   `bf16-true`    `adamw`           `False`    `auto`                        `-1`        

  Even larger model          `blip2-flan-t5-xxl`   `16/4`   `bf16-true`    `adamw`           `False`    `auto`                        `-1`        expected OOM

  XXL + smallest batch       `blip2-flan-t5-xxl`   `1/1`    `bf16-true`    `adamw`           `False`    `auto`                        `-1`        expected OOM

  Optimizer without state    `blip2-flan-t5-xxl`   `16/4`   `bf16-true`    `sgd`             `False`    `auto`                        `-1`        

  8-bit optimizer            `blip2-flan-t5-xxl`   `2/2`    `bf16-true`    `adam_8bit`       `False`    `auto`                        `-1`        

  Activation checkpointing   `blip2-flan-t5-xxl`   `2/2`    `bf16-true`    `adam_8bit`       `True`     `auto`                        `-1`        

  CPU offload (DeepSpeed)    `blip2-flan-t5-xxl`   `16/4`   `bf16-true`    `deepspeed_cpu`   `False`    `deepspeed_stage_2_offload`   `2`         
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

And this table summarizes the PEFT experiments.

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Experiment   Model                 bs/acc   precision   optim     act_ckpt   strategy   max_steps   Notes
  ------------ --------------------- -------- ----------- --------- ---------- ---------- ----------- ------------------------------------------------------------------------
  LoRA         `blip2-flan-t5-xxl`   `32/2`   `32-true`   `adamw`   `False`    `auto`     `-1`        `lr=5e-6`, `num_train_samples=512`, `use_lora=True`, `use_qlora=False`

  QLoRA        `blip2-flan-t5-xxl`   `64/1`   `32-true`   `adamw`   `False`    `auto`     `-1`        `lr=5e-6`, `num_train_samples=512`, `use_lora=False`, `use_qlora=True`
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Experiment: Baseline

As a baseline, let's try two epochs of fine-tuning Blip-2 using OPT-2.7b, an LLM with 2.7 billion parameters. When using Blip-2 with OPT-2.7b, the combined model will have 3.7 billion parameters.

Set `cfg` in `fine-tune-blip.py` to this baseline:

``` python
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

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

This run is expected to fail with OOM.

In the logs, we should see that parameters alone take close to 14 GB at full precision (32-bit). Then,

-   After the forward pass, activation memory spikes.
-   After backward pass, activation memory is freed, but now gradients are saved in memory.
-   After the optimizer step, optimizer state is allocated, memory jumps again, and the run goes OOM.

### Experiment: Reduced batch size

What if we reduce the batch size?

In `cfg`, change:

-   `"batch_size": 32` -\> `"batch_size": 16`

Leave all other values the same as the baseline.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

This run should now fit in memory, because with a smaller batch size, we need substantially less memory for activations.

Make a note of the training time and memory, which is printed at the end of the training job.

### Experiment: Gradient accumulation

By using gradient accumulation to "step" only after a few "micro batches", we can train with a larger effective "global" batch size, with minimal effect on the memory required.

In `cfg`, change:

-   `"accumulate_grad_batches": 1` -\> `"accumulate_grad_batches": 4`

Keep `batch_size` at `16` and leave all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

With gradient accumulation, we will see more output lines per step, simply because there are multiple forward passes and backward passes before the optimizer step.

Make a note of the training time and memory. We should see memory similar to the previous run.

You may notice that the loss after two epochs is different from the previous run - since the effective batch size has changed, we should scale the learning rate by the same amount to have a similar total step size over an epoch.

In the `cfg`, increase the learning rate by 4x (`"lr": 5e-6` -\> `"lr": 2e-5`), and run again:

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

### Experiment: Reduced precision

With a "bf16" format for numbers, instead of "float32", we can further reduce the memory required, although this representation is less precise.

In `cfg`, change:

-   `"precision": "32-true"` -\> `"precision": "bf16-true"`

Keep all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of the training time and memory, which is printed at the end of the training job.

This job should run much faster, and use much less memory, but it may also have higher loss than the previous run.

### Experiment: Mixed precision

With mixed precision, we get back some of the lost precision in the results, at the cost of additional memory.

In `cfg`, change:

-   `"precision": "bf16-true"` -\> `"precision": "bf16-mixed"`

Keep all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of the training time and memory, which is printed at the end of the training job.

You may notice that this job is faster than the equivalent full precision job, with comparable loss after two epochs.

### Experiment: Larger model

We've gained so much GPU memory back with these techniques, we can even train a larger model. Let's switch to a larger BLIP-2 model while keeping `bf16-true` precision:

In `cfg`, change:

-   `"model_name": "Salesforce/blip2-opt-2.7b"` -\> `"Salesforce/blip2-opt-6.7b"`
-   `"batch_size": 16` (keep at 16)
-   `"accumulate_grad_batches": 4` (keep at 4)
-   `"precision": "bf16-mixed"` -\> `"bf16-true"`

Leave all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of the training time and memory, which is printed at the end of the training job.

### Experiment: Even larger model

Let us try an even larger model. We will change out the LLM from `opt-6.7b` (6.7B parameters) to `flan-t5-xxl` (about 11B parameters).

In `cfg`, change:

-   `"model_name": "Salesforce/blip2-opt-6.7b"` -\> `"Salesforce/blip2-flan-t5-xxl"`

Leave all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

This run is expected to fail with OOM.

### Experiment: Even larger model with even smaller batch size

Even if we reduce to the smallest possible batch size, the `flan-t5-xxl` model is still too large for us to train on this GPU.

In `cfg`, change:

-   `"batch_size": 16` -\> `"batch_size": 1`
-   `"accumulate_grad_batches": 4` -\> `"accumulate_grad_batches": 1`

Leave all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

This run will OOM, too.

### Experiment: Optimizer without state (SGD)

The previous XXL runs fail partly because optimizer state takes a lot of memory. Next, we keep the XXL model but switch to an optimizer with no state.

In `cfg`, change:

-   `"batch_size": 1` -\> `"batch_size": 16`
-   `"accumulate_grad_batches": 1` -\> `"accumulate_grad_batches": 4`
-   `"optim": "adamw"` -\> `"optim": "sgd"`

Leave all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of training time and memory. Compare `Optim` memory with the previous run.

While we have freed up enough memory to train the XXL model, the result may not be great - the loss after 2 epochs may be poor.

### Experiment: Low-memory optimizer alternative (`adam_8bit`)

Another option is 8-bit Adam, which keeps optimizer state in reduced precision.

In `cfg`, change:

-   `"optim": "sgd"` -\> `"optim": "adam_8bit"`
-   `"batch_size": 16` -\> `"batch_size": 2`
-   `"accumulate_grad_batches": 4` -\> `"accumulate_grad_batches": 2`

Keep all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of training time and memory.

### Experiment: Activation checkpointing

Another way to reduce memory usage (at the cost of compute) is with activation checkpointing.

In `cfg`, change:

-   `"act_ckpt": False` -\> `"act_ckpt": True`

Keep all other values the same as the previous experiment.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of training time and memory, especially the activation memory.

We will see that we have saved on the activation memory, but at the cost of much slower training.

### Experiment: CPU offload with DeepSpeed

Finally, another way to save memory at the cost of compute is to use DeepSpeed CPU offload to move optimizer state off the GPU.

In `cfg`, change:

-   `"model_name": "Salesforce/blip2-flan-t5-xxl"`
-   `"precision": "bf16-true"`
-   `"optim": "deepspeed_cpu"`
-   `"act_ckpt": False`
-   `"strategy": "deepspeed_stage_2_offload"`
-   `"batch_size": 16`
-   `"accumulate_grad_batches": 4`
-   `"max_steps": 2`

This resets us to the "Even larger model" settings and then adds CPU offload. "Stage 2" here refers to ZeRO stage 2 - offloading optimizer state and gradients.

We set the maximum number of steps to 2 in this case, because training with CPU offload will be *so* slow - we really don't want to let it run to the end of 2 epochs.

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference):

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip.py
```

Make a note of training time and memory. We should see lower GPU memory pressure, with extra overhead from CPU offload.

In this run, the memory breakdown prints optimizer memory separately for GPU (`Optim GPU`) and CPU (`Optim CPU`). With DeepSpeed offload, a large fraction of optimizer state should move to CPU, so we expect `Optim CPU` to be high while `Optim GPU` stays much smaller.

### Experiment: Parameter efficient fine tuning

If we are only fine-tuning, not training a model from scratch, we can also consider LoRA and QLoRA. Let's try it first with our XXL model.

We are going to use the `fine-tune-blip-lora.py` script for our PEFT experiments, so open that and note

Set `cfg` in `fine-tune-blip-lora.py` to this LoRA configuration:

``` python
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

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference). Also check the `[cfg]` output near the top of the run to confirm the script is using the config you intended.

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip-lora.py
```

The memory required is much smaller! Earlier, we saw that full fine-tuning went OOM on this model in this configuration, but now we can fit it easily. Although the base model weights are still loaded (they are used in the forward/backward pass), there is only a small number of trainable parameters (just the adapter weights), so the gradients are much smaller and the optimizer state is much smaller.

We can also further reduce the memory required by quantizing the base model weights:

Set `cfg` in `fine-tune-blip-lora.py` to this QLoRA configuration:

``` python
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

After updating the config and saving the script, run the following cell, and also watch the `nvtop` output as it runs (take a screenshot of `nvtop` for later reference). Also check the `[cfg]` output near the top of the run to confirm the script is using the config you intended.

``` bash
# runs in the Jupyter service on node-llm-single
python fine-tune-blip-lora.py
```

When you have finished, download this notebook - which includes the output of each experiment stage - from the Jupyter environment for later reference.


<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>