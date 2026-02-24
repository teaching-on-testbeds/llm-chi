
::: {.cell .markdown}

# Large-scale model training on Chameleon - multi GPU

In this tutorial, we will practice fine-tuning a large language model. We will try two different strategies that distribute training across multiple GPUs:

* DDP
* FSDP

To run this experiment, you should have already created an account on Chameleon, and become part of a project. 

:::

::: {.cell .markdown .gpu-a100}

You must also have added your SSH key to the CHI@UC site (to use a 4x A100 GPU) or KVM@TACC site (to use a 4x H100 GPU).

:::


::: {.cell .markdown}

## Experiment topology 

In this experiment, we will deploy a single instance with four GPUs. We have "tuned" this experiment for two specific GPU types:

* 4x A100 with 80 GB VRAM (available on most `gpu_a100_pcie` bare metal instances at CHI@UC)
* or 4x H100 with 94GB VRAM (available in the `g1.h100.pci.4` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

:::


::: {.cell .markdown .gpu-a100}

You are currently viewing the A100 version of the instructions, but H100 instructions are also available at [index_h100](index_h100).

:::

::: {.cell .markdown .gpu-h100}

You are currently viewing the H100 version of the instructions, but A100 instructions are also available at [index_a100](index_a100).

:::
