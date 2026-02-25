# Large-scale model training on Chameleon - multi GPU

In this tutorial, we will practice fine-tuning a large language model. We will try two different strategies that distribute training across multiple GPUs:

-   DDP
-   FSDP

To run this experiment, you should have already created an account on Chameleon, and become part of a project.

## Experiment topology

In this experiment, we will deploy a single instance with four GPUs. We have "tuned" this experiment for two specific GPU types:

-   4x A100 with 80 GB VRAM (available on most `gpu_a100_pcie` bare metal instances at CHI@UC)
-   or 4x H100 with 94GB VRAM (available in the `g1.h100.pci.4` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

You are currently viewing the H100 version of the instructions, but A100 instructions are also available at [index_a100](index_a100).

## Create a lease

To use a GPU instance on Chameleon, we must reserve it in advance. GPU instances are much more in-demand than other resource types, and so we typically cannot make a reservation "on the spot" to use one.

We can use the OpenStack graphical user interface, Horizon, to reserve a GPU in advance. To access this interface,

-   from the [Chameleon website](https://chameleoncloud.org/hardware/)
-   click "Experiment" \> "KVM@TACC"
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

Reserve a 2 hr 50 minute block on a node with four H100 GPUs. This flavor is named `g1.h100.pci.4` on KVM@TACC.

-   On the left side, click on "Reservations" \> "Leases", and then click on "Flavor Calendar". In the "Node type" drop down menu, change the type to `g1.h100.pci.4` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC, so you will need to convert to your local time zone.
-   Once you have identified a 2 hr 50 minute block in UTC time that has GPU availability and works for you in your local time zone, make a note of the start and end time of the time you will try to reserve. (Note that if you mouse over a point on the graph, a pop up will show you the exact time.)
-   Then, on the left side, click on "Leases" again and then "Create Lease":
    -   set the "Name" to `llm_multi_netID`, replacing `netID` with your actual net ID.
    -   set the start date and time in UTC
    -   modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
    -   Click "Next".
-   On the "Flavors" tab,
    -   check the "Reserve Flavors" box
    -   let "Number of Instances for Flavor" be 1
    -   and click "Select" next to `g1.h100.pci.4`
    -   then click "Next".
-   Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time and some more details about the instance "flavor" you have reserved.

At the beginning of your lease time, continue with `2_create_server.ipynb`.

Before you begin, open this experiment on Trovi:

-   Use this link: [Large-scale model training on Chameleon](https://trovi.chameleoncloud.org/dashboard/artifacts/bd06bd6d-d94f-4297-ad5d-c9b7e1f02575) on Trovi
-   Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it.

Inside the `llm-chi` directory, open the `multi` subdirectory. You will see several notebooks - look for the one titled `2_create_server.ipynb`. Open this notebook and continue there.

## Bring up a GPU server

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

``` python
# run in Chameleon Jupyter environment
from chi import server, context, lease, network
import chi, os, time

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
```

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

``` python
# run in Chameleon Jupyter environment
l = lease.get_lease(f"llm_multi_netID")
l.show()
```

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook sets up the instance and the experiment environment, which cumulatively takes a while - bringing up the instance can take some time, and building the container image can take some time.

But, you can be mostly hands-off in this stage. You can save time by clicking on this cell, then selecting Run \> Run Selected Cell and All Below from the Jupyter menu.

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image.

The default boot disk for instances at KVM@TACC is a little small for large model training, so we will first create a larger boot volume (200 GiB) from that image, then boot the server from that volume.

``` python
# run in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this suffix
server_name = f"node-llm-multi-{username}"

try:
    s = server.get_server(server_name)
    print(f"Server {server_name} already exists. Skipping create.")
except Exception:
    os_conn = chi.clients.connection()
    cinder_client = chi.clients.cinder()

    images = list(os_conn.image.images(name="CC-Ubuntu24.04-CUDA"))
    image_id = images[0].id

    boot_vol = cinder_client.volumes.create(
        name=f"boot-vol-llm-multi-{username}",
        size=200,
        imageRef=image_id,
    )

    while True:
        boot_vol = cinder_client.volumes.get(boot_vol.id)
        if boot_vol.status == "available":
            break
        if boot_vol.status in ["error", "error_restoring", "error_extending"]:
            raise RuntimeError(f"Boot volume provisioning failed with status {boot_vol.status}")
        time.sleep(10)

    bdm = [{
        "boot_index": 0,
        "uuid": boot_vol.id,
        "source_type": "volume",
        "destination_type": "volume",
        "delete_on_termination": True,
    }]

    server_from_vol = os_conn.compute.create_server(
        name=server_name,
        flavor_id=server.get_flavor_id(l.get_reserved_flavors()[0].name),
        block_device_mapping_v2=bdm,
        networks=[{"uuid": os_conn.network.find_network("sharednet1").id}],
    )

    os_conn.compute.wait_for_server(server_from_vol)
    s = server.get_server(server_name)
```

We need security groups to allow SSH and Jupyter access.

``` python
# run in Chameleon Jupyter environment
security_groups = [
  {'name': "allow-ssh", 'port': 22, 'description': "Enable SSH traffic on TCP port 22"},
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"}
]
```

``` python
# run in Chameleon Jupyter environment
for sg in security_groups:
  secgroup = network.SecurityGroup({
      'name': sg['name'],
      'description': sg['description'],
  })
  secgroup.add_rule(direction='ingress', protocol='tcp', port=sg['port'])
  secgroup.submit(idempotent=True)
  s.add_security_group(sg['name'])

print(f"updated security groups: {[sg['name'] for sg in security_groups]}")
```

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
s.execute("git clone --branch h100 --single-branch https://github.com/teaching-on-testbeds/llm-chi")
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


## Build and start container for "Multiple GPU" section

Let's build the container image that we are going to use for this lab from `multi/docker/Dockerfile`. It may take 10-15 minutes to build this container image.

You may view this Dockerfile in our Github repository: [multi/docker/Dockerfile](https://github.com/teaching-on-testbeds/llm-chi/blob/main/multi/docker/Dockerfile).

This image starts from the Jupyter Pytorch CUDA12 stack and adds the pieces we need for this lab:

-   `nvtop` for NVIDIA GPU monitoring
-   ML Python libraries: notably, Lightning, Transformers and related libraries, and BitsAndBytes

``` python
# run in Chameleon Jupyter environment
s.execute("docker build -t llm-jupyter:latest ~/llm-chi/multi/docker")
```

and get it running:

``` python
# run in Chameleon Jupyter environment
s.execute("docker run --rm -d -p 8888:8888 -v /home/cc/llm-chi/multi/workspace:/home/jovyan/work --gpus all --name jupyter llm-jupyter:latest")
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

## Train a large model on multiple GPUs

In this section, we will practice strategies for training a large model using distributed processes across multiple GPUs. This section requires a host with 4x GPUs.

After completing this section, we should understand the effect of

-   distributed data parallelism
-   and learning-rate scaling with larger world size

on a large model training job.

Make sure that we can see the GPUs inside the container:

``` bash
# runs in the Jupyter service on node-llm-multi
nvidia-smi
```

Throughout these experiments, we will monitor GPU utilization with `nvtop`. Open a terminal in Jupyter (File \> New \> Terminal), then run:

``` bash
# runs in the Jupyter service on node-llm-multi
nvtop
```

Keep this running while training.

Before running the training script, download and unpack the dataset snapshot that we will use in this lab.

``` bash
# runs in the Jupyter service on node-llm-multi
cd ~/work
mkdir -p data
wget -O data/gourmetgram_caption.tar.gz "https://nyu.box.com/shared/static/g3qw3g5j7l8dkvyf02a9afuuo9grs3g3.gz"
mkdir -p data/gourmetgram_caption
tar -xzf data/gourmetgram_caption.tar.gz -C data/gourmetgram_caption --strip-components=1
```

The training script now reads the dataset from `./data/gourmetgram_caption`.

### Experiment 1: Single-GPU baseline on the multi-GPU node

We will start with a baseline for single-GPU performance before turning on distributed training.

Set `cfg` in `fine-tune-blip.py` to:

``` python
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

``` bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```

As it runs, note in `nvtop` that only one GPU is used. For GPU 0, GPU utilization is high and memory utilization is also high, while the other GPUs have zero utilization.

Also note that in the list of processes, there is a single process running on device 0.

Take a screenshot of this `nvtop` display while the script is running, for later reference.

When the training run finishes, note the training time and the memory summary printed by the script.

### Experiment 2: DDP with same batch settings and scaled LR

Now, we will repeat the same experiment with DDP across 4 GPUs, while keeping the same per-device batch settings.

With DDP, each GPU processes its own batch, so effective global batch is 4x larger than Experiment 1.

We scale the learning rate by 4x to keep the update magnitude more comparable.

DDP also allocates gradient communication buckets that are about the same order as total parameter size. Even with `gradient_as_bucket_view=True`, we still see this extra bucket memory, so we use the smaller model here.

In `cfg`, change:

-   `"devices": 1` -\> `"devices": 4`
-   `"strategy": "auto"` -\> `"strategy": "ddp"`
-   `"lr": 2e-5` -\> `"lr": 8e-5`

Leave all other values the same. Then, run:

``` bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```

Note that it may take a minute or two for the training job to start.

As it runs, note in `nvtop` that four GPUs are used, all with high utilization, and that four processes are listed. Take a screenshot of this `nvtop` display while the script is running, for later reference.

Check the memory logs printed by the Python script. Note that in distributed training, the `Other` value is a residual (`allocated - params - grads - optim`), that includes non-activation allocations such as communication buffers and collective buckets.

When the training run finishes, note the training time and memory summary printed by the script.

Compare this run with Experiment 1:

-   per-GPU memory will be larger than in single-GPU. Per-device batch settings are unchanged, but we also have memory overhead for communication buffers and collective buckets.
-   total throughput may or may not improve, because now four GPUs are working, but we also have communication overhead

### Experiment 3: FSDP

Now we will switch from DDP to FSDP, while keeping the same batch settings and learning rate as Experiment 2.

FSDP shards model states across GPUs, so it can reduce per-GPU memory pressure.

In our script, we define a set of layer classes and let Lightning FSDP auto-wrap those layers. Here, "wrap" means replacing each matching layer module with an FSDP-managed version of that module, so its parameters, gradients, and optimizer states can be sharded across GPUs instead of kept as full copies on every GPU.

``` python
_blip2_layer_cls = {Blip2EncoderLayer, Blip2QFormerLayer, OPTDecoderLayer}
```

and then the strategy passed to the Lightning `Trainer` is:

``` python
FSDPStrategy(auto_wrap_policy=_blip2_layer_cls)
```

In `cfg`, change:

-   `"strategy": "ddp"` -\> `"strategy": "fsdp"`

Leave all other values the same as Experiment 2.

Then, run:

``` bash
# runs in the Jupyter service on node-llm-multi
python fine-tune-blip.py
```

As it runs, note in `nvtop` that four GPUs are used, and pay attention to memory usage on each GPU. Compare this run with Experiment 2.

When the training run finishes, note the training time and memory summary printed by the script.


<hr>

<small>Questions about this material? Contact Fraida Fund</small>

<hr>

<small>This material is based upon work supported by the National Science Foundation under Grant No. 2230079.</small>

<small>Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.</small>