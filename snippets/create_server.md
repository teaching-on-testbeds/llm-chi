

::: {.cell .markdown}

Before you begin, open this experiment on Trovi:

* Use this link: [Large-scale model training on Chameleon](https://chameleoncloud.org/experiment/share/39a536c6-6070-4ccf-9e91-bc47be9a94af) on Trovi
* Then, click “Launch on Chameleon”. This will start a new Jupyter server for you, with the experiment materials already in it.

You will see several notebooks inside the `llm-chi` directory - look for the one titled `1_create_server.ipynb`. Open this notebook and continue there.

:::

::: {.cell .markdown}

## Bring up a GPU server

At the beginning of the lease time, we will bring up our GPU server. We will use the `python-chi` Python API to Chameleon to provision our server. 

We will execute the cells in this notebook inside the Chameleon Jupyter environment.

Run the following cell, and make sure the correct project is selected:

:::

::: {.cell .code}
```python
from chi import server, context, lease
import os

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@UC")
```
:::

::: {.cell .markdown}

Change the string in the following cell to reflect the name of *your* lease (**with your own net ID**), then run it to get your lease:

:::

::: {.cell .code}
```python
l = lease.get_lease(f"llm_netID") # or llm_single_netID, or llm_multi_netID
l.show()
```
:::

::: {.cell .markdown}

The status should show as "ACTIVE" now that we are past the lease start time.

The rest of this notebook can be executed without any interactions from you, so at this point, you can save time by clicking on this cell, then selecting Run > Run Selected Cell and All Below from the Jupyter menu.  

As the notebook executes, monitor its progress to make sure it does not get stuck on any execution error, and also to see what it is doing!

:::

::: {.cell .markdown}

We will use the lease to bring up a server with the `CC-Ubuntu24.04-CUDA` disk image. (Note that the reservation information is passed when we create the instance!) This will take up to 10 minutes.

:::


::: {.cell .code}
```python
username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-llm-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-CUDA"
)
s.submit(idempotent=True)
```
:::

::: {.cell .markdown}

Note: security groups are not used at Chameleon bare metal sites, so we do not have to configure any security groups on this instance.

:::

::: {.cell .markdown}

Then, we'll associate a floating IP with the instance, so that we can access it over SSH.

:::

::: {.cell .code}
```python
s.associate_floating_ip()
```
:::

::: {.cell .code}
```python
s.refresh()
s.check_connectivity()
```
:::

::: {.cell .markdown}

In the output below, make a note of the floating IP that has been assigned to your instance (in the "Addresses" row).

:::

::: {.cell .code}
```python
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
s.execute("git clone https://github.com/teaching-on-testbeds/llm-chi")
```
:::


::: {.cell .markdown}

## Set up Docker with NVIDIA container toolkit

To use common deep learning frameworks like Tensorflow or PyTorch, we can run containers that have all the prerequisite libraries necessary for these frameworks. Here, we will set up the container framework.

:::

::: {.cell .code}
```python
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
:::


::: {.cell .markdown}

In the following cell, we will verify that we can see our NVIDIA GPUs from inside a container, by passing `--gpus-all`. (The `-rm` flag says to clean up the container and remove its filesystem when it finishes running.)

:::


::: {.cell .code}
```python
s.execute("docker run --rm --gpus all ubuntu nvidia-smi")
```
:::

::: {.cell .markdown}

Let's pull the actual container images that we are going to use, 

* For the "Single GPU" section: a Jupyter notebook server with PyTorch and CUDA libraries
* For the "Multiple GPU" section: a PyTorch image with NVIDIA developer tools, which we'll need in order to install DeepSpeed

:::

::: {.cell .markdown}

## Pull and start container for "Single GPU" section

Let's pull the container:

:::

::: {.cell .code}
```python
s.execute("docker pull quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1")
```
:::


::: {.cell .markdown}

and get it running:

:::

::: {.cell .code}
```python
s.execute("docker run -d -p 8888:8888 --gpus all --name torchnb quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.5.1")
```
:::


::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Finally, run

:::

::: {.cell .code}
```python
s.execute("docker logs torchnb")
```
:::

::: {.cell .markdown}

Look for the line of output in the form:

```
http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

and copy it for use in the next section.

:::

::: {.cell .markdown}

You will continue working from the next notebook! But, if you are planning to do the "Multiple GPU" section in the same lease, run the following cells too, to let the container image for the "Multiple GPU" section get pulled in the background. This image takes a loooooong time to pull, so it's important to get it started and leave it running while you are working in your other tab on the "Single GPU" section.

:::


::: {.cell .markdown}

## Pull container for "Multiple GPU" section


:::


::: {.cell .code}
```python
s.execute("docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
```
:::



::: {.cell .markdown}

and let's also install some software on the host that we'll use in the "Multiple GPU" section:

:::


::: {.cell .code}
```python
s.execute("sudo apt update; sudo apt -y install nvtop")
```
:::



