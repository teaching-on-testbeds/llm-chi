# Large-scale model training on Chameleon

In this tutorial, we practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

* gradient accumulation
* reduced precision
* parameter efficient fine tuning
* distributed training across multiple GPUs and with CPU offload

Follow along at [Large-scale model training on Chameleon](https://teaching-on-testbeds.github.io/llm-chi/).

This lab has two parts:

* `single/`: single-GPU large-model training, requires an A100 80GB or H100 GPU
* `multi/`: multi-GPU large-model training, requires a 4x A100 80GB or 4x H100 GPU

---

This material is based upon work supported by the National Science Foundation under Grant No. 2230079.
