---
title: Week 1
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

**% Completion:** 10% (Orientation and Implementation Preparation)

**Key Milestones Achieved This Week:**
- Established a shared understanding of how FMS structures model integration and compile execution
- Outlined the end-to-end workflow for bringing Granite Speech into the FMS, from model load to compiling to benchmarking

**Deliverables Submitted:**
- Integration plan

---

## 3. Tasks Completed This Week

| Task | Description | Outcome / Results | Responsible Member |
|------|-------------|-------------------|-------------------|
| Codebase Reviews | Analyzed how FMS registers models and handles torch.compile | Clarified where Granite Speech hooks into existing interfaces | All team members |
| Granite Model Study | Reviewed model components and input/output structure | Understood data flow needed for FMS integration | All team members |
| Integration Mapping | Traced dependencies between FMS, optimizer, and testing repos | Defined how we'll reuse existing benchmark and compile utilities | All team members |
| Implementation Plan | Converted proposal goals into practical development milestones | Ready to start coding integration layer next week | All team members |

---

## 4. Plans for Next Week

| Planned Task | Expected Outcome | Assigned To |
|--------------|------------------|-------------|
| Start integration implementation | Set up Granite Speech inside FMS so the model can be loaded and run in eager mode | Geonsik |
| Prepare compile hooks | Enable initial torch.compile testing to observe graph break behavior | Aneesh |
| Set up benchmark template | Run a small latency and memory profiling test on dummy data | Zach |
| Begin documentation draft | Record integration steps and compile observations for internal tracking | In Keun |

---

## 5. Challenges / Blockers

| Issue | Description | Impact | Proposed Solution / Support Needed |
|-------|-------------|--------|-----------------------------------|
| N/A | No major blockers this week | N/A | N/A |

---

## 6. Individual Contributions (This Week)

| Student Name | Key Contributions | Hours Contributed | Notes |
|--------------|------------------|-------------------|-------|
| Aneesh | Codebase review and torch.compile analysis | 8 | Focused on compile execution flow |
| Geonsik | FMS model registration analysis | 8 | Identified integration points |
| In Keun | Documentation framework setup | 8 | Prepared tracking templates |
| Zach | Benchmark utilities exploration | 8 | Reviewed profiling tools |

---

## 7. Feedback / Requests from Supervisors

**Specific Questions / Feedback Needed:**
- Review and confirm that our understanding of the project scope and planned next steps are correct

**Areas Where Guidance is Required:**
- Guidance on which Granite Speech checkpoint to use for baseline integration
- Clarification on the Granite Vision implementation in FMS that was mentioned as reference

---

## 8. Appendix (Optional)

**Key Files to Modify/Create:**

1. `fms/models/granite_speech.py`
   - Main model implementation

2. `fms/modules/audio_encoder.py`
   - Audio encoding module

3. `fms/modules/audio_projector.py`
   - Audio-to-text feature mapping

4. `fms/models/hf/granite_speech/`
   - HuggingFace adapter

5. `tests/models/test_granite_speech.py`
   - Test suite

**Granite-Speech Architecture Diagram:**

```
Input: Text + Audio
    ├── Text Input ──→ Tokenizer (Granite) ──┐
    │                                        │
    └── Audio Input ──→ Audio Encoder        │
                        (Whisper-style)      │
                        - Conv layers        │
                        - Transformer ───→ Audio Projector
                                          (Learnable Map)
                                          Audio → Token
                                               │
                                               ├──→ Granite Blocks
                                               │    (Multimodal ATN)
                                               │    - Self-Attn
                                               │    - Cross-Attn
                                               │    - FFN
                                               │
                                               └──→ Output Head
                                                    (Text/Audio Gen)
```

**Repository Links or Pull Requests:**
- Integration planning document (internal)

**Relevant Papers / References:**
- [Granite Speech Paper](https://arxiv.org/pdf/2505.08699)
- [Granite Speech on HuggingFace](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- [IBM Granite Documentation](https://www.ibm.com/granite/docs/models/speech)

---

*Report Generated: 2025-11-11*
