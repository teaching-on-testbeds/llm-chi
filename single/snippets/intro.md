
::: {.cell .markdown}

# Large-scale model training on Chameleon - single GPU

In this tutorial, we will practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

* gradient accumulation
* reduced precision
* parameter efficient fine tuning

To run this experiment, you should have already created an account on Chameleon, and become part of a project. 

:::

::: {.cell .markdown .gpu-a100}

You must also have added your SSH key to the CHI@UC site (to use an A100 GPU) or KVM@TACC site (to use an H100 GPU).

:::


::: {.cell .markdown}

## Experiment topology 

In this experiment, we will deploy a single instance with a GPU. We have "tuned" this experiment for two specific GPU types:

* an A100 with 80 GB VRAM (available on most `compute_gigaio` bare metal instances at CHI@UC)
* or an H100 with 94GB VRAM (available in the `g1.h100.pci.1` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

:::


::: {.cell .markdown .gpu-a100}

You are currently viewing the A100 version of the instructions, but H100 instructions are also available at [index_h100](index_h100).

:::

::: {.cell .markdown .gpu-h100}

You are currently viewing the H100 version of the instructions, but A100 instructions are also available at [index_a100](index_a100).

:::
