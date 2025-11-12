---
title: Week 3
hide_title: true
---

## 1. Project Overview

**Project Title:** Granite Speech Integration in the Foundation Model Stack (FMS)

**Faculty Supervisor:** Dr. Kaoutar El Maghraoui, Dr. Rashed Bhatti

**Student Team Members:**

- Aneesh
- Geonsik
- In Keun
- Zach

**Project Objective:** Integrate the Granite Speech model into IBM's Foundation Model Stack so it can run end-to-end under torch.compile, evaluate its performance against eager execution, and document what makes a speech model compile-efficient inside FMS.

---

## 2. Overall Progress Summary

**% Completion:** 40%

**Key Milestones Achieved This Week:**

- Completed baseline FMS Conformer encoder implementation (~515 lines)
- Implemented all Conformer components: Feed-Forward, Attention with Shaw's relative positional embeddings, Convolution module with GLU gating
- Created comprehensive test suite for validation
- Established baseline CPU performance metrics for FMS Conformer
- Identified architectural differences between FMS Conformer and HuggingFace Wav2Vec2Conformer

**Deliverables Submitted:**

- FMS Conformer implementation (conformer.py)
- Comprehensive test suite (test_conformer.py)
- Baseline performance tests (test_conformer_simple.py)
- Demo script and documentation

---

## 3. Tasks Completed This Week

| Task                           | Description                                                                                        | Outcome / Results                                                                  | Responsible Member |
| ------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------ |
| Conformer Skeleton & Interface | Set up TDD skeleton with complete docstrings, type hints, and test infrastructure                  | Created 515-line skeleton with all components defined                              | In Keun            |
| Feed-Forward Module            | Implemented ConformerFeedForward with LayerNorm, Linear layers, Activation, and Dropout            | Shape tests pass, no NaN/Inf, gradients flow correctly                             | Geonsik            |
| Attention Module               | Implemented ConformerAttention with Shaw's relative positional embeddings and multi-head attention | Precomputed 5000x5000 attention distance matrix with learnable position embeddings | Aneesh             |
| Convolution Module             | Implemented ConformerConvModule with depthwise separable convolution and GLU gating                | Pointwise expansion, GLU, Depthwise conv, BatchNorm, Pointwise compression         | Zach               |
| Conformer Block                | Integrated all components with correct residual scaling (0.5x for FF, 1.0x for attention/conv)     | Follows Conformer paper architecture                                               | Geonsik            |
| Conformer Encoder              | Implemented full encoder with input projection and block stacking                                  | Handles variable sequence lengths (50-1000), no temporal downsampling              | Geonsik            |
| Validation Tests               | Created comprehensive test suite for components, shapes, numerical stability, gradients            | All tests pass: no NaN/Inf, gradients flow, variable lengths work                  | In Keun            |
| Baseline Performance Test      | Benchmarked FMS Conformer on CPU                                                                   | Small config (4L-256H): 49.96 ± 4.90 ms, 40.04 samples/sec                         | Aneesh             |
| Documentation                  | Created quick start guide, implementation summary, and README                                      | Complete documentation for usage and testing                                       | Zach               |

---

## 4. Plans for Next Week

| Planned Task                           | Expected Outcome                                                       | Assigned To |
| -------------------------------------- | ---------------------------------------------------------------------- | ----------- |
| Implement Speech Projector (Q-Former)  | Audio-to-text fusion with temporal downsampling in speech-projector.py | Team        |
| Profile FMS Conformer hotspots         | Identify performance bottlenecks using torch.profiler                  | Team        |
| Apply torch.compile() to FMS Conformer | Measure compilation speedup on CPU and GPU                             | Team        |
| Run GPU benchmarks                     | Compare FMS Conformer performance on H200 GPU                          | Aneesh      |

---

## 5. Challenges / Blockers

| Issue                                     | Description                                                                                                                                                           | Impact                                                                       | Proposed Solution / Support Needed                                                                                                                                                                         |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HuggingFace Architectural Incompatibility | FMS Conformer expects pre-extracted log-mel features (batch, seq_len, 80), while HF Wav2Vec2Conformer expects raw audio waveforms with built-in CNN feature extractor | Cannot perform direct performance comparison with HuggingFace implementation | Document as architectural design decision. FMS Conformer is designed as standalone encoder for pre-extracted features (Granite Speech encoder component). Focus on standalone FMS performance benchmarking |
| Baseline performance not optimized        | FMS Conformer is unoptimized baseline (~50ms for small config on CPU)                                                                                                 | Need optimization before production use                                      | Next steps: torch.compile(), Flash Attention 2, kernel fusion                                                                                                                                              |

---

## 6. Individual Contributions (This Week)

| Student Name | Key Contributions                                                                                    | Hours Contributed | Notes                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------- |
| Geonsik      | Implemented Feed-Forward, Conformer Block, and Conformer Encoder components                          | 12 hours          | Complete baseline implementation ready for optimization  |
| In Keun      | Set up TDD skeleton with docstrings and type hints, created comprehensive test suite for validation  | 8 hours           | Solid foundation for iterative development               |
| Aneesh       | Implemented Attention module with Shaw's relative positional embeddings, ran baseline CPU benchmarks | 10 hours          | Core attention mechanism with relative position encoding |
| Zach         | Implemented Convolution module with GLU gating, wrote documentation                                  | 8 hours           | Depthwise separable convolution with proper gating       |

---

## 7. Feedback / Requests from Supervisors

**Specific Questions / Feedback Needed:**

- Should we prioritize optimizing the FMS Conformer encoder further, or shift focus to Speech Projector and decoder integration?
- Are there specific datasets you recommend for end-to-end validation?

**Areas Where Guidance is Required:**

- Architectural design for Q-Former implementation in Speech Projector
- Priority: encoder optimization vs Speech Projector vs decoder optimization vs end-to-end integration

---

## 8. Appendix

### FMS Conformer Architecture

**Implementation Details:**

```
ConformerEncoder
├── Input Projection: 80 log-mel features → hidden_dim
├── Conformer Blocks (stacked × num_layers)
│   ├── Feed-Forward 1 (0.5x residual)
│   ├── Multi-Head Attention with Shaw's Relative Positional Embeddings (1.0x residual)
│   ├── Convolution Module with GLU Gating (1.0x residual)
│   ├── Feed-Forward 2 (0.5x residual)
│   └── Post Layer Normalization
└── Output: (batch, seq_len, hidden_dim)
```

**Key Technical Decisions:**

1. Shaw's Relative Positional Embeddings: Precomputed 5000x5000 distance matrix, learnable embeddings
2. Depthwise Separable Convolution: Pointwise expansion, GLU gating, Depthwise conv, BatchNorm, Pointwise compression
3. Residual Connection Scaling: 0.5x for feed-forward modules, 1.0x for attention and convolution (per Conformer paper)
4. No Temporal Downsampling: Sequence length preserved through all blocks

### Baseline CPU Performance (FMS Conformer)

**Test Configuration:**

- Hardware: M3 MacBook (CPU only)
- Model: FMS Conformer Small (4L-256H)
- Input: batch_size=2, seq_length=100, features=80
- Precision: FP32
- Benchmark: 10 warmup iterations + 50 runs

**Results:**

| Configuration        | Parameters | Inference Time  | Throughput        | Status |
| -------------------- | ---------- | --------------- | ----------------- | ------ |
| Small (4L-256H)      | 7.4M       | 49.96 ± 4.90 ms | 40.04 samples/sec | Pass   |
| Variable seq_len=50  | 7.4M       | 13.70 ± 0.21 ms | 72.99 samples/sec | Pass   |
| Variable seq_len=100 | 7.4M       | 19.85 ± 8.45 ms | 50.38 samples/sec | Pass   |
| Variable seq_len=200 | 7.4M       | 18.28 ± 0.48 ms | 54.71 samples/sec | Pass   |
| Variable seq_len=500 | 7.4M       | 40.25 ± 0.33 ms | 24.84 samples/sec | Pass   |

**Validation Results:**

- No NaN/Inf values in outputs
- Gradients flow correctly through all layers
- Variable sequence lengths (50, 100, 200, 500, 1000) all pass
- Output shapes match expected dimensions

### FMS Conformer vs HuggingFace: Architectural Comparison

**Key Difference: Input Modality**

| Aspect             | FMS Conformer                                       | HuggingFace Wav2Vec2Conformer                  |
| ------------------ | --------------------------------------------------- | ---------------------------------------------- |
| Input              | Pre-extracted log-mel features (batch, seq_len, 80) | Raw audio waveform (batch, samples)            |
| Feature Extraction | External (pre-processing step)                      | Internal CNN feature extractor (7 conv layers) |
| Architecture Scope | Encoder only                                        | Feature extractor + Encoder                    |
| Use Case           | Standalone encoder for Granite Speech               | End-to-end speech recognition                  |

**Configuration Mapping (Encoder Layers):**

| FMS Conformer                 | HuggingFace Wav2Vec2Conformer     |
| ----------------------------- | --------------------------------- |
| hidden_dim                    | hidden_size                       |
| num_layers                    | num_hidden_layers                 |
| num_heads                     | num_attention_heads               |
| feedforward_mult × hidden_dim | intermediate_size                 |
| conv_kernel_size              | conv_kernel (tuple)               |
| dropout                       | attention_dropout, hidden_dropout |

**Implication for Testing:**

Direct performance comparison is not possible due to different input modalities. Focus areas:

1. Standalone FMS Conformer performance benchmarking
2. Component-level validation (shapes, gradients, numerical stability)
3. Future: Compare with other spectrogram-based encoders (e.g., Whisper encoder)

### Repository Links or Pull Requests:

- Conformer Implementation: `foundation-model-stack/fms/models/conformer.py`
- Validation Tests: `foundation-model-stack/tests/models/test_conformer.py`
- Baseline Performance Tests: `foundation-model-stack/tests/models/hf_equivalence/test_conformer_simple.py`
- Branch: `granite-speech`

### Relevant Papers / References:

- [Conformer Paper](https://arxiv.org/abs/2005.08100) - Gulati et al., 2020
- [Shaw's Relative Positional Embeddings](https://arxiv.org/abs/1803.02155) - Shaw et al., 2018
- [Granite Speech Paper](https://arxiv.org/pdf/2505.08699)
- [HuggingFace Wav2Vec2Conformer](https://huggingface.co/docs/transformers/model_doc/wav2vec2-conformer)

---

_Report Generated: 2025-11-12_
