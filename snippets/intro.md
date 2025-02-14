
::: {.cell .markdown}

# Large-scale model training on Chameleon

In this tutorial, we will practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

* gradient accumulation
* reduced precision
* parameter efficient fine tuning
* distributed training across multiple GPUs and with CPU offload

To run this experiment, you should have already created an account on Chameleon, and become part of a project. You must also have added your SSH key to the CHI@UC site.

:::

::: {.cell .markdown}

## Experiment topology 

In this experiment, we will deploy a single bare metal instance with specific NVIDIA GPU capabilities.

* In the "Single GPU" section: To use `bfloat16` in our experiments with reduced precision, we need a GPU with [NVIDIA CUDA compute capability](https://developer.nvidia.com/cuda-gpus) 8.0 or higher. For example, some Chameleon nodes have A100 or A30 GPUs, which have compute capability 8.0. If we use a V100, with compute capability 7.0, the `bfloat16` capability is actually emulated in software.
* In the "Multiple GPU" section: To practice distributed training with multiple GPUs, will request a node with 4 GPUs. 


We can browse Chameleon hardware configurations for suitable node types using the [Hardware Browser](https://chameleoncloud.org/hardware/). 

For example, to find nodes with 4x GPUs: if we expand "Advanced Filters", check the "4" box under "GPU count", and then click "View", we can identify some suitable node types: `gpu_a100_pcie`, `gpu_a100_nvlink`, `gpu_v100`, or `gpu_v100_nvlink` at CHI@UC. (The [NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/)-type nodes have a high-speed interconnect between GPUs.)

(We will avoid `p100`-based node types for this experiment, because the P100 has less GPU RAM, and less compute capability.)

:::

::: {.cell .markdown}

## Create a lease

:::

::: {.cell .markdown}

To use bare metal resources on Chameleon, we must reserve them in advance. We can use the OpenStack graphical user interface, Horizon, to submit a lease for an A100 or V100 node at CHI@UC. To access this interface,

* from the [Chameleon website](https://chameleoncloud.org/hardware/)
* click "Experiment" > "CHI@UC"
* log in if prompted to do so
* check the project drop-down menu near the top left (which shows e.g. “CHI-XXXXXX”), and make sure the correct project is selected.

:::

::: {.cell .markdown}

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
  
Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease.

:::

::: {.cell .markdown}

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

:::

::: {.cell .markdown}

Since you will need the full lease time to actually execute your experiment, you should read *all* of the experiment material ahead of time in preparation, so that you make the best possible use of your GPU time.

:::

::: {.cell .markdown}

At the beginning of your lease time, you will continue with the next step, in which you bring up a bare metal instance!

:::
