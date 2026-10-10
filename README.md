# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Utkarsh's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/utkarsh530.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Named-Dimension Batched Attention Scores | Compute batched multi-head query-key scores by contracting only the head-width dimension. | https://www.tensortonic.com/problems/cs336-l02-einsum-attention-scores |
| Transformer Training FLOP Estimator | Estimate one training step from forward matrix multiplications and a supplied forward attention cost. | https://www.tensortonic.com/problems/cs336-l02-training-flop-estimator |
| Patchify a Batch of Images | Split PyTorch image batches into a row-major grid of non-overlapping flattened Vision Transformer patches. | https://www.tensortonic.com/problems/cs336-l17-vision-transformer-patchify |
| 2D Sinusoidal Positional Embedding | Build the 2D sin-cos positional embedding used by ViT-MAE, DINOv2, and similar models. | https://www.tensortonic.com/problems/cv-2d-sincos-pos-embed |
| ViT Patchify | Split image batches into row-major non-overlapping patches and flatten each patch in spatial-then-channel order. | https://www.tensortonic.com/problems/cv-patchify |
| Layer Normalization | Implement Layer Normalization (Ba et al, 2016), the standard normalization technique in Transformers. | https://www.tensortonic.com/problems/dl-layer-normalization |
| Scaled Dot-Product Attention | Implement scaled dot-product attention, the fundamental building block of all Transformer architectures. | https://www.tensortonic.com/problems/dl-scaled-dot-product-attention |
| Implement Autoregressive Decoding with a KV Cache | Decode autoregressively with an append-only key-value cache so previously processed tokens are never recomputed. | https://www.tensortonic.com/problems/inference-autoregressive-kv-cache |
| Calculate KV Cache Memory for MHA, MQA, GQA, and MLA | Compute the total KV-cache memory, in bytes, for a full sequence under four attention variants: MHA, MQA, GQA, and MLA. | https://www.tensortonic.com/problems/inference-kv-cache-memory |
| Implement Sparse MoE Top-k Expert Routing | Select the top k highest-scoring experts per token and compute routing weights as a softmax over only those selected logits. | https://www.tensortonic.com/problems/inference-moe-top-k-routing |
| Implement Multi-Head Attention (MHA) | Split into h heads, run scaled dot-product attention per head with an optional causal mask, concatenate the heads, and apply an output projection. | https://www.tensortonic.com/problems/inference-multi-head-attention |
| Implement Scaled Dot-Product Attention | Implement batched scaled dot-product attention for self- and cross-attention with optional masks and stable softmax. | https://www.tensortonic.com/problems/inference-scaled-dot-product-attention |
| Scaled Dot-Product Attention | Implement the scaled dot-product attention mechanism from the Transformer architecture. | https://www.tensortonic.com/problems/la-scaled-attention |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Attention Mechanism from Scratch | Implement the scaled dot-product attention mechanism, a core building block of the Transformer architecture. | https://www.tensortonic.com/problems/pytorch-attention-from-scratch |
| Masked Causal Attention | Implement scaled dot-product attention with a causal mask that prevents each position from attending to future positions. | https://www.tensortonic.com/problems/pytorch-masked-causal-attention |
| KV Cache Append | Append one autoregressive decoding row to key and value caches in Triton without modifying other cache positions. | https://www.tensortonic.com/problems/triton-kv-append |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/utkarsh530)
<!-- tensortonic:end -->
