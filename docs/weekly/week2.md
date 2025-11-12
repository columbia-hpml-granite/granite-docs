---
title: Week 2
hide_title: true
---

## Project Overview

**Project Title:** Granite Speech Integration in the Foundation Model Stack (FMS)

**Faculty Supervisor:** Dr. Kaoutar El Maghraoui, Dr. Rashed Bhatti

**Student Team Members:**

- [Aneesh Durai](https://github.com/aneeshdurai)
- [Geonsik Moon](https://github.com/gsmoon97)
- [In Keun Kim](https://github.com/nearKim)
- [Zachary Zusin](https://github.com/zacharyzusin)

**Project Objective:** Integrate the Granite Speech model into IBM's Foundation Model Stack so it can run end-to-end under torch.compile, evaluate its performance against eager execution, and document what makes a speech model compile-efficient inside FMS.

---

## Overall Progress Summary

**% Completion:** 25%

**Key Milestones Achieved This Week:**

- Completed comprehensive profiling and performance analysis of Granite Speech model
- Discovered critical finding: Granite Speech is decoder-bound (decoder dominates end-to-end latency)
- Established baseline performance metrics across multiple configurations
- Implemented profiler infrastructure in `aiu_fms_testing_utils`

**Deliverables Submitted:**

- Performance profiling report
- Encoder-decoder latency breakdown analysis
- Profiler implementation in testing utilities

**Key Finding: Granite Speech is Decoder-Bound**

---

## Tasks Completed This Week

| Task                     | Description                                                       | Outcome / Results                                                          | Responsible Member |
| ------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------ |
| Performance Profiling    | Implemented comprehensive profiling framework for Granite Speech  | Profiler available in `aiu_fms_testing_utils/utils/torch_profiler.py`      | Zach               |
| Encoder-Decoder Analysis | Measured latency breakdown between encoder and decoder components | Encoder accounts for only 0.7-3.6% of total latency; decoder is bottleneck | Geonsik            |
| DType Experiments        | Compared bf16 vs fp16 precision modes                             | Minimal performance difference (~1ms variance); both viable options        | Aneesh             |
| Token Length Analysis    | Tested impact of max_new_tokens (64 vs 128) on throughput         | Longer sequences reduce tokens/s for 30s audio (274→163 tokens/s)          | In Keun            |
| Baseline Validation      | Validated profiler on HuggingFace Wav2Vec2-base                   | Confirmed 4.5% speedup with torch.compile (18.9→18.1ms)                    | Team               |

---

## Plans for Next Week

| Planned Task                           | Expected Outcome                                    | Assigned To |
| -------------------------------------- | --------------------------------------------------- | ----------- |
| Add encoder/projector timing breakdown | Detailed profiling of encoder subcomponents         | Zach        |
| Run batch-size sweeps                  | Understand batching impact on throughput and memory | Geonsik     |
| Export profiling results               | Generate visualizations for performance analysis    | Aneesh      |
| Begin FMS encoder integration          | Start implementing conformer blocks in FMS          | Team        |

---

## Challenges / Blockers

| Issue                     | Description                                   | Impact                                         | Proposed Solution / Support Needed                                                                |
| ------------------------- | --------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Decoder dominates latency | 96.4-99.3% of inference time spent in decoder | Encoder optimizations will have minimal impact | Shift focus to decoder optimization and investigate speculative decoding or KV cache improvements |
| Audio dataset selection   | Need appropriate datasets for testing         | Cannot validate on real-world scenarios        | Awaiting supervisor guidance on recommended datasets                                              |

---

## Appendix (Optional)

### Experimental Setup

**Hardware & Configuration:**

- **GPU:** NVIDIA H200
- **Model:** Granite Speech 3.3 8B (HuggingFace)
- **Audio Lengths:** 3, 10, 30 seconds
- **Precision Modes:** bf16, fp16
- **Token Lengths:** 64, 128
- **Compilation Modes:** eager, compile

### Encoder-Decoder Breakdown Results

![Chart](/img/weekly/week2/1.png)

| Audio (s) | Total (ms) | Encoder (ms) | Decoder (ms) | Encoder % | RTF   | Peak Mem (MB) |
| --------- | ---------- | ------------ | ------------ | --------- | ----- | ------------- |
| 3 s       | 489.4      | 17.7         | 471.7        | 3.6 %     | 0.163 | 17,353        |
| 10 s      | 532.3      | 18.2         | 514.1        | 3.4 %     | 0.053 | 17,376        |
| 30 s      | 2662.2     | 18.8         | 2643.4       | 0.7 %     | 0.089 | 17,430        |

**Key Observation:** Encoder time remains nearly constant (~18ms) regardless of audio length, while decoder time scales linearly with output length.

### DType Comparison (bf16 vs fp16)

![Chart](/img/weekly/week2/2.png)

| DType | Audio (s) | Latency (ms) | RTF   | Tokens/s | Peak Mem (MB) |
| ----- | --------- | ------------ | ----- | -------- | ------------- |
| bf16  | 3         | 497.0        | 0.166 | 120.7    | 34,652        |
| bf16  | 10        | 538.1        | 0.054 | 249.0    | 34,674        |
| bf16  | 30        | 2684.3       | 0.089 | 162.1    | 34,728        |
| fp16  | 3         | 498.8        | 0.166 | 120.3    | 34,652        |
| fp16  | 10        | 535.1        | 0.054 | 250.4    | 34,674        |
| fp16  | 30        | 2667.5       | 0.089 | 163.1    | 34,728        |

**Conclusion:** Negligible difference between bf16 and fp16 (~1-17ms variance across all tests).

### Token Length Comparison

![Chart](/img/weekly/week2/3.png)

| Setting              | Audio (s) | Latency (ms) | Tokens/s |
| -------------------- | --------- | ------------ | -------- |
| max_new_tokens = 64  | 3         | 495.6        | 121.1    |
|                      | 10        | 555.2        | 241.3    |
|                      | 30        | 1352.0       | 274.4    |
| max_new_tokens = 128 | 3         | 495.8        | 121.0    |
|                      | 10        | 537.2        | 249.5    |
|                      | 30        | 2673.6       | 162.7    |

**Observation:** For 30s audio, throughput drops significantly with longer generation (274→163 tokens/s).

### Baseline Validation (Wav2Vec2)

Validated profiler on HuggingFace Wav2Vec2-base model:

| Mode     | Latency (ms) | Throughput (samples/s) | GPU Mem (MB) |
| -------- | ------------ | ---------------------- | ------------ |
| Eager    | 18.9         | 52.9                   | 765          |
| Compiled | 18.1         | 55.3                   | 765          |

**Speedup:** 4.5% improvement with torch.compile

### Performance Visualization

**Encoder vs Decoder Latency by Audio Length:**

- 3s audio: 17.7ms encoder, 471.7ms decoder (Total: 489ms)
- 10s audio: 18.2ms encoder, 514.1ms decoder (Total: 532ms)
- 30s audio: 18.8ms encoder, 2643.4ms decoder (Total: 2662ms)

**Next Steps:**

- Add encoder/projector timing breakdown
- Run batch-size sweeps
- Export profiling results for visualization
- Begin FMS conformer block implementation

**Repository Links or Pull Requests:**

- Profiler implementation: `aiu_fms_testing_utils/utils/torch_profiler.py`
- Colab notebook with GPU setup and FMS dependencies

**Relevant Papers / References:**

- [Granite Speech Paper](https://arxiv.org/pdf/2505.08699)
- [Granite Speech on HuggingFace](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- [IBM Granite Documentation](https://www.ibm.com/granite/docs/models/speech)

---

_Report Generated: 2025-11-11_
