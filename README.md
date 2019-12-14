# neurpy
![pytorch](https://img.shields.io/static/v1?label=PyTorch&message=1.2.0&color=ec4d35)
![numpy](https://img.shields.io/static/v1?label=NumPy&message=1.16.3&color=158aca)

<p align="center">
  <img align="center" src="graphics/neurpy.png" width="300"/>
</p>

## Installation
To install the package, run the following commands:
```
git clone git@github.com:Liberty3000/neurpy.git
cd neurpy
pip install -e .
```
___
### Adversarial Networks
| Model | Reference | Link |
|-------|-----------|------|
| StyleGANv2 | Analyzing and Improving the Image Quality of StyleGAN | [arXiv:1912.04958](https://arxiv.org/abs/1912.04958) |
| StarGANv2 | StarGAN v2: Diverse Image Synthesis for Multiple Domains | [arXiv:1912.01865](https://arxiv.org/abs/1912.01865) |
| BigBiGAN | Large Scale Adversarial Representation Learning | [arXiv:1907.02544](https://arxiv.org/abs/1907.02544) |
| FUNIT | Few-Shot Unsupervised Image-to-Image Translation | [arXiv:1905.01723](https://arxiv.org/abs/1905.01723) |
| MSG-GAN | MSG-GAN: Multi-Scale Gradient GAN for Stable Image Synthesis | [arXiv:1903.06048](https://arxiv.org/abs/1903.06048) |
| StyleGAN | A Style-Based Generator Architecture for Generative Adversarial Networks | [arXiv:1812.04948](https://arxiv.org/abs/1812.04948) |
| ProGAN | Progressive Growing of GANs for Improved Quality, Stability, and Variation | [arXiv:1710.10196](https://arxiv.org/abs/1710.10196) |
| BigGAN | Large Scale GAN Training for High Fidelity Natural Image Synthesis | [arXiv:1809.11096](https://arxiv.org/abs/1809.11096) |
| StarGAN | StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation | [arXiv:1711.09020](https://arxiv.org/abs/1711.09020) |
| CycleGAN | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks | [arXiv:1703.10593](https://arxiv.org/abs/1703.10593) |
| pix2pix | Image-to-Image Translation with Conditional Adversarial Networks | [arXiv:1611.07004](https://arxiv.org/abs/1611.07004) |
| WGAN | Improved Training of Wasserstein GANs | [arXiv:1704.00028](https://arxiv.org/abs/1704.00028) |
| ACGAN | Conditional Image Synthesis With Auxiliary Classifier GANs | [arXiv:1610.09585](https://arxiv.org/abs/1610.09585) |
| DCGAN | Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks | [arXiv:1511.06434](https://arxiv.org/abs/1511.06434) |
| CGAN | Conditional Generative Adversarial Nets | [arXiv:1411.1784](https://arxiv.org/abs/1411.1784) |

### Autoencoders
| Model | Reference | Link |
|-------|-----------|------|
| AE      | Recent Advances in Autoencoder-Based Representation Learning   | [arXiv:1812.05069](https://arxiv.org/abs/1812.05069) |
| AAE     | Adversarial Autoencoders                                       | [arXiv:1511.05644](https://arxiv.org/abs/1511.05644) |
| VAE     | Auto-Encoding Variational Bayes                                | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)   |
| InfoVAE | InfoVAE: Information Maximizing Variational Autoencoders       | [arXiv:1706.02262](https://arxiv.org/abs/1706.02262) |
| VQ-VAE   | Neural Discrete Representation Learning                       | [arXiv:1711.00937](https://arxiv.org/abs/1711.00937) |
| PIONEER | Pioneer Networks: Progressively Growing Generative Autoencoder | [arXiv:1807.03026](https://arxiv.org/abs/1807.03026) |

### Flows
| Model | Reference | Link |
|-------|-----------|------|
| NAF | Neural Autoregressive Flows | [arXiv:1804.00779](https://arxiv.org/abs/1804.00779) |
| MAF | Masked Autoregressive Flow for Density Estimation | [arXiv:1705.07057](https://arxiv.org/abs/1705.07057) |
| IAF | Improving Variational Inference with Inverse Autoregressive Flow | [arXiv:1606.04934](https://arxiv.org/abs/1606.04934) |
| NICE | NICE: Nonlinear Independent Components Estimation | [arXiv:1410.8516](https://arxiv.org/abs/1410.8516) |
| MADE | Masked Autoencoder for Density Estimation | [arXiv:1502.03509](https://arxiv.org/abs/1502.03509) |
| MaCow | Masked Convolutional Generative Flow | [arXiv:1902.04208](https://arxiv.org/abs/1902.04208) |
| Glow | Glow: Generative Flow with Invertible 1×1 Convolutions | [arXiv:1807.03039](https://arxiv.org/abs/1807.03039) |
| RealNVP | Density Estimation using Real NVP | [arXiv:1605.08803](https://arxiv.org/abs/1605.08803) |
| IRN | Invertible Residual Networks | [arXiv:1811.00995](https://arxiv.org/abs/1811.00995) |
| PixelCNN | Conditional Image Generation with PixelCNN Decoders | [arXiv:1606.05328](https://arxiv.org/abs/1606.05328) |
| PixelRNN | Pixel Recurrent Neural Networks | [arXiv:1601.06759](https://arxiv.org/abs/1601.06759) |
| PixelSNAIL | PixelSNAIL: An Improved Autoregressive Generative Model | [arXiv:1712.09763](https://arxiv.org/abs/1712.09763) |

### Transformers
| Model | Reference | Link |
|-------|-----------|------|
| Transformer | Attention Is All You Need | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) |
| TransformerXL | Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | [arXiv:1901.02860](https://arxiv.org/abs/1901.02860) |
| XLNet | XLNet: Generalized Autoregressive Pretraining for Language Understanding | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) |
| XLM | Cross-lingual Language Model Pretraining | [arXiv:1901.07291](https://arxiv.org/abs/1901.07291) |
| CTRL | CTRL: A Conditional Transformer Language Model for Controllable Generation | [arXiv:1909.05858](https://arxiv.org/abs/1909.05858) |
| LevT | Levenshtein Transformer | [arXiv:1905.11006](https://arxiv.org/abs/1905.11006) |

### Neural Processes
| Model | Reference | Link |
|-------|-----------|------|
| NP | Neural Processes | [arXiv:1807.01622](https://arxiv.org/abs/1807.01622) |
| CNP | Conditional Neural Processes | [arXiv:1807.01613](https://arxiv.org/abs/1807.01613) |
| ANP | Attentive Neural Processes | [arXiv:1901.05761](https://arxiv.org/abs/1901.05761) |
| FNP | Functional Neural Processes | [arXiv:1906.08324](https://arxiv.org/abs/1906.08324) |
| RNP | Recurrent Neural Processes | [arXiv:1906.05915](https://arxiv.org/abs/1906.05915) |
