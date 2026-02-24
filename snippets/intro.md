# Large-scale model training on Chameleon

As a machine learning engineer at GourmetGram, you've previously developed a model that automatically categorizes uploaded food photos (e.g., Bread, Dessert, Soup). This is a relatively small model, and doesn't require any special training strategies for scale.

Your next challenge will be to build a model that generates **alt text** for images.

Alt text is a short description of an image that is embedded in the HTML alongside the image itself. It is important for accessibility and searchability: so that screen readers (used by those with visual impairments) and search engines can understand what the image shows.

You’ll be fine-tuning a [BLIP-2](https://arxiv.org/abs/2301.12597) model - a vision-language model that can understand both images and text. BLIP-2 (Bootstrapped Language-Image Pretraining) combines a visual encoder (which processes the image) and a large language model (which generates the caption). It's designed to generate natural-language descriptions of images, so it's a good choice for alt-text generation.  Don’t worry if you’re not familiar with vision-language models or large language models - all you need to know is that BLIP-2 is a powerful pre-trained model that can turn images into text.

BLIP-2 is available with different underlying language models of varying sizes and capabilities, such as FlanT5 (XL and XXL variants) or OPT (2.7B and 6.7B variants). Larger versions generally produce more fluent and detailed captions, but require more computational resources. Because of that, you’ll need to use some special training strategies to handle models at this scale efficiently.

You’ll fine-tune your model using a dataset or GourmetGram images that have user-contributed captions.

This exercise has two parts:

* `single/`: single-GPU workflows for fine-tuning large language models
* `multi/`: multi-GPU workflows for distributed large-model training

You can do either part independently. If possible, start with `single/` and then continue to `multi/`.
