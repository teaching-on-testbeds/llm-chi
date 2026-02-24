::: {.cell .markdown}

Before you begin, open this experiment on Trovi:

* Use this link: [Large-scale model training on Chameleon](https://chameleoncloud.org/experiment/share/39a536c6-6070-4ccf-9e91-bc47be9a94af) on Trovi
* Then, click "Launch on Chameleon". This will start a new Jupyter server for you, with the experiment materials already in it.

Inside the `llm-chi` directory, open the `single` subdirectory. You will see several notebooks - look for the one titled `2_create_server.ipynb`. Open this notebook and continue there.

:::

::: {.cell .markdown}

## Bring up a GPU server

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server.

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

:::

::: {.cell .code .gpu-a100}
```python
# run in Chameleon Jupyter environment
from chi import server, context, lease
import os

context.version = "1.0"
context.choose_project()
context.choose_site(default="CHI@UC")
```
:::

::: {.cell .code .gpu-h100}
```python
# run in Chameleon Jupyter environment
from chi import server, context, lease, network
import chi, os, time

context.version = "1.0"
context.choose_project()
context.choose_site(default="KVM@TACC")
```
:::

::: {.cell .markdown}

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
l = lease.get_lease(f"llm_single_netID")
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

:::

::: {.cell .markdown}

The rest of this notebook sets up the instance and the experiment environment, which cumulatively takes a while - bringing up the instance can take some time, and building the container image can take some time. 

But, you can be mostly hands-off in this stage. You can save time by clicking on this cell, then selecting Run > Run Selected Cell and All Below from the Jupyter menu.

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

:::

::: {.cell .markdown}

We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image.

:::

::: {.cell .markdown .gpu-a100}

Bare metal instances can take much longer than VM instances to bring up, and the `gigaio` nodes in particular take even longer - up to 30 minutes. 

So if it takes a while to build the instance, you just need to be patient - as long as it does not show the instance in `ERROR` state, it's working as expected.

:::


::: {.cell .code .gpu-a100}
```python
# run in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this suffix
s = server.Server(
    f"node-llm-single-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown .gpu-a100}

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

:::


::: {.cell .markdown .gpu-h100}

The default boot disk for instances at KVM@TACC is a little small for large model training, so we will first create a larger boot volume (200 GiB) from that image, then boot the server from that volume.

:::

::: {.cell .code .gpu-h100}
```python
# run in Chameleon Jupyter environment
username = os.getenv('USER') # all exp resources will have this suffix

os_conn = chi.clients.connection()
cinder_client = chi.clients.cinder()

images = list(os_conn.image.images(name="CC-Ubuntu24.04-CUDA"))
image_id = images[0].id

boot_vol = cinder_client.volumes.create(
    name=f"boot-vol-llm-single-{username}",
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
    name=f"node-llm-single-{username}",
    flavor_id=server.get_flavor_id(l.get_reserved_flavors()[0].name),
    block_device_mapping_v2=bdm,
    networks=[{"uuid": os_conn.network.find_network("sharednet1").id}],
)

os_conn.compute.wait_for_server(server_from_vol)
s = server.get_server(f"node-llm-single-{username}")
```
:::

::: {.cell .markdown .gpu-h100}

We need security groups to allow SSH and Jupyter access.

:::

::: {.cell .code .gpu-h100}
```python
# run in Chameleon Jupyter environment
security_groups = [
  {'name': "allow-ssh", 'port': 22, 'description': "Enable SSH traffic on TCP port 22"},
  {'name': "allow-8888", 'port': 8888, 'description': "Enable TCP port 8888 (used by Jupyter)"}
]
```
:::

::: {.cell .code .gpu-h100}
```python
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
:::

::: {.cell .markdown}

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.refresh()
s.show(type="widget")
```
:::


::: {.cell .markdown}

## Retrieve code and notebooks on the instance

Now, we can use `python-chi` to execute commands on the instance, to set it up. We'll start by retrieving the code and other materials on the instance.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("git clone https://github.com/teaching-on-testbeds/llm-chi")
```
:::


::: {.cell .markdown}

## Set up Docker with NVIDIA container toolkit

To use common deep learning frameworks like Tensorflow or PyTorch, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")
s.execute("docker run hello-world")
```
:::

::: {.cell .markdown}

We will also install the NVIDIA container toolkit, with which we can access GPUs from inside our containers.

:::

::: {.cell .code}
```python
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
:::


::: {.cell .markdown}

In the following cell, we will verify that we can see our NVIDIA GPUs from inside a container, by passing `--gpus all`. (The `-rm` flag says to clean up the container and remove its filesystem when it finishes running.)

:::


::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("docker run --rm --gpus all ubuntu nvidia-smi")
```
:::

::: {.cell .markdown}


:::

::: {.cell .markdown}

## Build and start container for "Single GPU" section

Let's build the container image that we are going to use for this lab from `single/docker/Dockerfile`. It may take 10-15 minutes to build this container image.

You may view this Dockerfile in our Github repository: [single/docker/Dockerfile](https://github.com/teaching-on-testbeds/llm-chi/blob/main/single/docker/Dockerfile).

This image starts from the Jupyter Pytorch CUDA12 stack and adds the pieces we need for this lab:

* CUDA toolkit 12.8 so we can build/install packages that require CUDA tooling
* `nvtop` for NVIDIA GPU monitoring
* ML Python libraries: notably, Lightning, Transformers and related libraries, and BitsAndBytes
* DeepSpeed for CPU offload experiments with larger models

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("docker build -t llm-jupyter:latest ~/llm-chi/single/docker")
```
:::


::: {.cell .markdown}

and get it running:

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("docker run --rm -d -p 8888:8888 -v /home/cc/llm-chi/single/workspace:/home/jovyan/work --gpus all --name jupyter llm-jupyter:latest")
```
:::

::: {.cell .markdown}

To access the Jupyter service, we will need its randomly generated secret token (which secures it from unauthorized access). We'll get this token by running `jupyter server list` inside the `jupyter` container on the `node-llm-<username>` instance:

:::

::: {.cell .code}
```python
# run in Chameleon Jupyter environment
s.execute("docker exec jupyter jupyter server list")
```
:::

::: {.cell .markdown}

Look for a line like

```
http://localhost:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Paste this into a browser tab, but in place of `localhost`, substitute the floating IP assigned to your instance, to open the Jupyter notebook interface.

Continue with the notebook located inside that workspace.

:::
