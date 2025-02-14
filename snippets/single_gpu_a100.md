
::: {.cell .markdown}

## Train a large model on a single GPU

In this section, we will practice strategies for training a large model on a single GPU. After completing this section, you should understand the effect of

* batch size
* gradient accumulation
* reduced precision/mixed precision
* parameter efficient fine tuning

on a large model training job.

:::


::: {.cell .markdown}

### Open the notebook on Colab

We should have already started a notebook server in a container on a Chameleon GPU host, and set up an SSH tunnel to this notebook server. Now, we will open this notebook in Google Colab and connect it to the runtime that you have in Chameleon. This is a convenient way to work, because the notebook and its outputs will be saved automatically in your Google Drive.

* Use this button to open the notebook in Colab: <a target="_blank" href="https://colab.research.google.com/github/teaching-on-testbeds/llm-chi/blob/main/workspace/2_single_gpu_a100.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
* Click "File > Save a Copy in Drive" to save it in your own Google Drive.
* Next to the "Connect" button in the top right, there is a &#9660; symbol. Click on this symbol to expand the menu, and choose "Connect to a local runtime".
* Paste the `http://127.0.0.1:8888/lab?token=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX` you copied earlier into this space, and choose "Connect".

Alternatively, if you prefer not to use Colab (or can't, for some reason): just put the URL you copied earlier into your browser to open the Jupyter interface directly.

:::



::: {.cell .markdown}

### Prepare LitGPT

For this tutorial, we will fine-tune an [TinyLlama](https://arxiv.org/abs/2401.02385) or [OpenLLaMA](https://github.com/openlm-research/open_llama) large language model using [`litgpt`](https://github.com/Lightning-AI/litgpt). LitGPT is a convenient wrapper around many PyTorch Lightning capabilities that makes it easy to fine-tune a GPU using a "recipe" defined in a YAML file. (We'll also try the Python API for LitGPT in the "Multiple GPU" section of this tutorial.)

Our focus will be exclusively on comparing the time and memory requirements of training jobs under different settings - we will completely ignore the loss of the fine-tuned model, and we will make some choices to reduce the overall time of our experiment (to fit in a short Chameleon lease) that wouldn't make sense if we really needed the fine-tuned model (e.g. using a very small fraction of the training data).

First, install LitGPT:

:::

::: {.cell .code}
```python
!pip install 'litgpt[all]'==0.5.7 'lightning<2.5.0.post0'
```
:::


::: {.cell .markdown}


then, download the foundation models:

:::


::: {.cell .code}
```python
!litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```
:::


::: {.cell .code}
```python
!litgpt download openlm-research/open_llama_3b
```
:::

::: {.cell .code}
```python
!litgpt download openlm-research/open_llama_7b
```
:::


::: {.cell .code}
```python
!litgpt download openlm-research/open_llama_13b
```
:::



::: {.cell .markdown}


Also, get the "recipes" that we will use for LLM fine-tuning. Using the file browser on the left side, look at the contents of the "config" directory.

:::


::: {.cell .code}
```python
!git clone https://github.com/teaching-on-testbeds/llm-chi/
!mv llm-chi/workspace/config .
```
:::



::: {.cell .markdown}

### Baseline

As a baseline, let's try an epoch of fine-tuning the TinyLlama-1.1B, using full precision and a batch size of 32:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 32
```
:::

::: {.cell .markdown}

This will fail because the training job won't fit in our 80GB GPU memory.

:::


::: {.cell .markdown}

### Reduced batch size

But with a smaller batch size, it fits easily:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 8 --train.micro_batch_size 8
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Gradient accumulation

By using gradient accumulation to "step" only after a few "micro batches", we can train with a larger effective "global" batch size, with minimal effect on the memory required:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Reduced precision

With a "brain float16" format for numbers, instead of "float32", we can further reduce the memory required, although this representation is less precise:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::





::: {.cell .markdown}

### Mixed precision

With mixed precision, we get back some of the lost precision in the results, at the cost of some additional memory and time:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-mixed
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Larger model - 3b

We've gained so much GPU memory back with these techniques, we can even train a larger model. Let's switch from the 1.1B to the 3B model:
:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/open-llama-3b-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --precision bf16-true
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::

::: {.cell .markdown}

### Larger model - 7b

If we reduce the batch size again, we can even train a 7b model:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/open-llama-7b-full.yaml --train.global_batch_size 16 --train.micro_batch_size 4 --precision bf16-true
```
:::

::: {.cell .markdown}

Make a note of the training time and memory, which is printed at the end of the training job.

:::


::: {.cell .markdown}

### Larger model - 13b

Even with the smallest possible batch size, we can't train a 13B model:

:::


::: {.cell .code}
```python
!litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true
```
:::

::: {.cell .markdown}

this will fail with an "out of memory" error. But, if we switch from the Adam optimizer (which has two state values per parameter) to SGD, we can train a 13B model. It's *verrrrry* slow, though, so we won't even train it for a full epoch - just 25 "steps", so we can get an idea of the memory required:

:::

::: {.cell .code}
```python
!litgpt finetune_full --config config/open-llama-13b-full.yaml --train.global_batch_size 1 --train.micro_batch_size 1 --precision bf16-true --optimizer SGD --train.max_steps 25
```
:::



::: {.cell .markdown}

### Parameter efficient fine tuning

If we are only fine-tuning, not training a model from scratch, we can also consider LoRA and QLoRA. Let's try it first with our 1.1B model:

:::


::: {.cell .code}
```python
!litgpt finetune --config config/tiny-llama-lora.yaml
```
:::

::: {.cell .markdown}

The memory required is *shockingly* small! We can see it with our 3B and 7B models, too:
:::

::: {.cell .code}
```python
!litgpt finetune --config config/open-llama-3b-lora.yaml
```
:::

::: {.cell .code}
```python
!litgpt finetune --config config/open-llama-7b-lora.yaml
```
:::

::: {.cell .markdown}

We can also further reduce the memory required with quantization:


:::

::: {.cell .code}
```python
!litgpt finetune --config config/open-llama-7b-lora.yaml --quantize bnb.nf4
```
:::



::: {.cell .markdown}

Even the 13B model can be trained quickly with minimal memory required, using LoRA:

:::

::: {.cell .code}
```python
!litgpt finetune --config config/open-llama-13b-lora.yaml
```
:::

