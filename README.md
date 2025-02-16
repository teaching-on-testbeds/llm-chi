In this tutorial, we practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

* gradient accumulation
* reduced precision
* parameter efficient fine tuning
* distributed training across multiple GPUs and with CPU offload

Follow along at [Large-scale model training on Chameleon](https://teaching-on-testbeds.github.io/llm-chi/).

Note: this tutorial requires advance reservation of specific hardware! Either:

* one node with 4x A100 80GB: `gpu_a100_pcie` at CHI@UC for a 3-hour block, or
* a 2-hour block on a node with 4x A100 80GB or 4x V100: `gpu_a100_pcie` or `gpu_v100` AND a 2-hour block on a node with 1x A100 80GB: `compute_gigaio` at CHI@UC

---

This material is based upon work supported by the National Science Foundation under Grant No. 2230079.
