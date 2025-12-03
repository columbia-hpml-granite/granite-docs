---
title: Week 6
hide_title: true
---

## Project Overview

**Project Title:** Granite Speech Integration in Foundation Model Stack (FMS)

**Faculty Supervisors:**
- Dr. Kaoutar El Maghraoui
- Dr. Rashed Bhatti

**Student Team Members:**
- [Aneesh Durai](https://github.com/aneeshdurai)
- [Geonsik Moon](https://github.com/gsmoon97)
- [In Keun Kim](https://github.com/nearKim)
- [Zachary Zusin](https://github.com/zacharyzusin)

**Project Objective:** Integrate IBM's Granite Speech 3.3 8B model into the Foundation Model Stack to enable end-to-end execution under `torch.compile`, focusing on achieving HuggingFace equivalence and preparing for final integration.

---

## Overall Progress Summary

**% Completion:** 85%

**Key Milestones Achieved This Week:**
- ✅ Fixed HuggingFace alignment issues in Conformer Encoder (16 → 10 blocks)
- ✅ Completed HF to FMS Equivalence Test for Granite Speech (183 lines)
- ✅ Fixed critical multimodal forward path and attention masking bugs
- ✅ Identified remaining gaps for final integration (Feature Extractor, Processor, LoRA)
- ✅ Added temporary HF end-to-end test reference code

**Deliverables Submitted:**
- Updated Conformer configuration (10 conformer blocks)
- HF-to-FMS equivalence test suite
- Bug fixes for multimodal forward path
- Gap analysis documentation
- Reference HF end-to-end test code (temporary)

---

## Tasks Completed This Week

| Task | Description | Outcome / Results | Responsible Member |
|------|-------------|-------------------|-------------------|
| **Conformer Block Configuration Update** | Changed ConformerEncoder from 16 → 10 conformer blocks to match HF implementation | Successfully aligned with HF: [fms/models/conformer.py:1](fms/models/conformer.py:1), [fms/models/granite_speech.py:1](fms/models/granite_speech.py:1) | Geonsik Moon |
| **HF to FMS Equivalence Testing** | Created comprehensive test suite to validate FMS implementation against HuggingFace | 183 lines of equivalence tests in [tests/models/hf_equivalence/test_granite_speech.py:1](tests/models/hf_equivalence/test_granite_speech.py:1) | Aneesh Durai |
| **Projector TODO Completion** | Refactored and completed remaining TODO items from Week 5 projector implementation | 398 lines refactored (183 additions, 215 deletions) in [fms/modules/projector.py:1](fms/modules/projector.py:1) | Aneesh Durai |
| **Multimodal Forward Path Bug Fixes** | Fixed attention masking, input validation, and audio embedding alignment issues | Improved safety with ValueError messages, enforced mutual exclusion of input_ids/inputs_embeds, required input_features_mask | Aneesh Durai |
| **Gap Analysis & Documentation** | Analyzed FMS vs HF implementation to identify missing components | Documented 5 critical gaps: Config fields, Query shape, Query init, Window processing, Output reshape | In Keun Kim |
| **Temporary HF Reference Code** | Added reference HF end-to-end test code (wav → txt) for comparison purposes | Added with `tmp_` prefix for reference (not tested due to runtime), to be deleted after integration | In Keun Kim |
| **HF Discrepancy Fixes** | Fixed discrepancies between FMS and HuggingFace implementations | Improved compatibility and alignment with HF behavior | In Keun Kim |

---

## Technical Details

### 1. Conformer Configuration Update

**Changes Made:**
- Updated `ConformerEncoder` from 16 conformer blocks to 10 blocks to match HuggingFace implementation
- Ensures proper alignment for weight conversion and equivalence testing

**Files Modified:**
- [fms/models/conformer.py:1](fms/models/conformer.py:1) - Configuration update
- [fms/models/granite_speech.py:1](fms/models/granite_speech.py:1) - Integration update

### 2. Multimodal Forward Path Improvements

**Key Bug Fixes:**
```python
# Fixed in granite_speech.py:
- Pass attention_mask through to GraniteHeadless in GraniteSpeech.forward
- Harden get_merged_audio_embeddings:
  - Enforce alignment between audio positions and audio_features
  - Surface clear ValueError on mismatch instead of relying on masked_scatter failure
- Tighten input validation in forward:
  - Enforce mutual exclusion of input_ids / inputs_embeds
  - Require input_features_mask when input_features is provided
```

**Impact:** Makes GraniteSpeech implementation safer and closer to expected HuggingFace behavior without changing public APIs or model configuration.

### 3. Identified Implementation Gaps

Based on detailed comparison between FMS and HuggingFace implementations:

#### **Critical Config Issues:**
1. **Missing `window_size` field** in `SpeechProjectorConfig`
   - HF uses `window_size=15` for windowed attention
   - FMS: Field not defined in config

2. **Query Shape Mismatch**
   - HF: `self.query = nn.Parameter(torch.zeros(1, self.num_queries, config.projector_config.hidden_size))`
   - FMS: `self.query_embeds = nn.Parameter(torch.zeros(config.num_queries, config.encoder_dim))`
   - Missing first dimension of `1`

3. **Query Initialization**
   - HF: `self.query.data.normal_(mean=0.0, std=1.0)`
   - FMS: Missing initialization (stays zeros)
   - Need to add: `nn.init.normal_(self.query_embeds, mean=0.0, std=1.0)`

#### **Window-Based Processing Gap:**
```python
# HF Implementation:
nblocks = math.ceil(seq_len / self.window_size)
pad = nblocks * self.window_size - seq_len
hidden_states = F.pad(hidden_states, (0, 0, 0, pad), "constant", 0)
hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, dim)

# FMS Implementation: Missing - uses global attention on all frames
query_states = self.query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
```

#### **Output Reshape Missing:**
```python
# HF Implementation:
query_output.last_hidden_state.view(batch_size, nblocks * self.num_queries, -1)

# FMS Implementation: Missing - returns fixed shape
return projected_states  # Always (batch, 32, decoder_dim)
```

### 4. Integration Gaps Analysis

**✅ GOOD - Matching Components:**
- Audio feature extraction: `get_audio_features()` method
- Embedding merging: `get_merged_audio_embeddings()` with masked scatter
- Forward pass logic: Proper handling of input_ids, input_features, and inputs_embeds

**❌ CRITICAL GAPS:**

1. **Missing Feature Extractor**
   - HF has: `GraniteSpeechFeatureExtractor` - converts raw audio to mel spectrograms
   - FMS has: Nothing - expects pre-processed features
   - **Impact:** HIGH - Users must manually extract features before using the model

2. **Missing Processor**
   - HF has: `GraniteSpeechProcessor` - handles text expansion of `<|audio|>` tokens
   - FMS has: Nothing - users must manually expand tokens
   - **Impact:** HIGH - Easy to get wrong token count vs. audio feature count

3. **No LoRA Adapter Support**
   - HF has: Conditional LoRA activation during audio processing
   - FMS has: No LoRA integration
   - **Impact:** MEDIUM - Model won't match HF performance without LoRA adapters

4. **Missing KV Cache Handling**
   - HF has: Proper `past_key_values` management for efficient generation
   - FMS has: Basic support but not fully tested
   - **Impact:** MEDIUM - Generation may be slower

---

## Plans for Next Week

| Planned Task | Expected Outcome | Assigned To |
|--------------|------------------|-------------|
| **Fix Projector Config Issues** | Add `window_size` field, fix query shape, add proper initialization | Geonsik Moon |
| **Implement Window-Based Processing** | Add windowed attention processing to match HF behavior | Geonsik Moon |
| **Implement Feature Extractor** | Create audio-to-mel-spectrogram conversion module | In Keun Kim |
| **Implement Processor Module** | Create token expansion handler for `<|audio|>` tokens | Zachary Zusin |
| **LoRA Integration Research** | Investigate LoRA adapter requirements and integration approach | Aneesh Durai |
| **KV Cache Testing** | Validate and optimize KV cache handling for generation | Aneesh Durai |
| **End-to-End Integration Testing** | Run complete wav → txt pipeline tests | All |

---

## Challenges / Blockers

| Issue | Description | Impact | Proposed Solution / Support Needed |
|-------|-------------|--------|-----------------------------------|
| **HF E2E Test Runtime** | End-to-end test with HuggingFace model (wav → txt) takes too long to complete | Cannot validate against HF baseline quickly | Added reference code for manual comparison; consider smaller test audio samples |
| **Window Processing Complexity** | Window-based attention requires significant refactoring of projector forward pass | May introduce bugs if not carefully implemented | Thorough unit testing at each step; compare intermediate outputs with HF |
| **LoRA Integration Scope** | Unclear what level of LoRA support is required for equivalence | May affect performance validation | Need guidance on whether full LoRA parity is required or optional |
| **Feature Extractor Scope** | Need to determine if feature extractor should be in FMS or separate preprocessing | Affects API design decisions | Clarify whether FMS should handle raw audio or expect preprocessed features |

---

## Individual Contributions (This Week)

| Student Name | Key Contributions | Hours Contributed | Notes |
|--------------|------------------|-------------------|-------|
| **In Keun Kim** | Identified and documented HF-FMS implementation gaps; added temporary HF reference code for comparison | 8 | Gap analysis critical for planning final integration |
| **Aneesh Durai** | Created HF-to-FMS equivalence test suite (183 lines); completed projector TODO refactoring (398 lines); fixed multimodal forward path bugs | 10 | Equivalence testing ensures FMS-HF parity |
| **Geonsik Moon** | Updated Conformer configuration from 16 to 10 blocks; validated alignment with HF implementation | 6 | Critical configuration fix for weight conversion |
| **Zachary Zusin** | Fixed HF discrepancies; validated test suite execution | 5 | Ensured test infrastructure is working correctly |

---

## Feedback / Requests from Supervisors

**Specific Questions / Feedback Needed:**
- Should we implement LoRA support for full equivalence, or is it optional for this integration?
- What is the expected scope for the Feature Extractor - should it be part of FMS or a separate preprocessing step?
- Do we need to support raw audio input (requiring Feature Extractor) or can we assume preprocessed mel spectrograms?
- Should window-based processing be configurable or always enabled to match HF?

**Areas Where Guidance is Required:**
- Priority ordering for remaining implementation gaps (Feature Extractor vs Processor vs LoRA vs KV Cache)
- Performance benchmarking requirements - what metrics should we target?
- Timeline expectations for final integration and testing

---

## Appendix

**Branch:** `granite-speech` (commit `2a1d558`)

**Files Modified/Created This Week:**
| File | Lines Changed | Description |
|------|---------------|-------------|
| [fms/models/conformer.py:1](fms/models/conformer.py:1) | +14, -4 | Updated ConformerEncoder config (16 → 10 blocks) |
| [fms/models/granite_speech.py:1](fms/models/granite_speech.py:1) | +54, -52 | Fixed multimodal forward path and attention masking |
| [fms/modules/projector.py:1](fms/modules/projector.py:1) | +183, -215 | Completed TODO refactoring |
| [tests/models/hf_equivalence/test_granite_speech.py:1](tests/models/hf_equivalence/test_granite_speech.py:1) | +183 | HF-to-FMS equivalence test suite |
| `tests/models/test_*_hf.py` (temporary) | N/A | Reference HF end-to-end test code (not tested) |

**Test Execution Status:**
```bash
# From Week 5:
$ pytest tests/models/test_granite_speech.py tests/models/test_conformer_hf.py tests/modules/test_projector_hf.py -v

================== 68 passed, 1 skipped in 174.28s (0:02:54) ===================
```

**Key Commits Since November 19, 2024:**
- `5c5b8ee` - Update config for ConformerEncoder (16 → 10 conformer blocks)
- `ad977ae` - [tmp] Add temporary HuggingFace e2e test code for reference
- `b9c4f7f` - [tmp] Fix discrepancies with HuggingFace
- `150f197` - Fix GraniteSpeech multimodal forward path and attention masking
- `af6a532` - Create HF to FMS Equivalence Test for Granite Speech
- `02fc0ed` - Completed projector.py TODO's from last week's Projector implementation

**Repository Links:**
- Main Repository: [foundation-model-stack (granite-speech branch)](https://github.com/columbia-hpml-granite/foundation-model-stack/tree/granite-speech)

**Relevant Implementation References:**
- [HuggingFace Granite Speech Model](https://huggingface.co/ibm-granite/granite-3.3-8b-speech)
- [Original Granite Speech Paper](https://arxiv.org/abs/2501.xxxxx) (if available)

---

*Report Generated: 2025-12-03*
