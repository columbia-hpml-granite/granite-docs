---
title: Week 7
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

**% Completion:** 90%

**Key Milestones Achieved This Week:**

- Completed `GraniteSpeechFeatureExtractor` and `GraniteSpeechProcessor` implementations
- Added comprehensive E2E tests with real audio from LibriSpeech dataset (Not tested for now)
- Implemented LoRaAdapter feature for Granite Speech model
- Migrated existing tests from HF and fixed all test bugs - all tests now passing or skipping (GPU required tests)

**Deliverables Submitted:**

- Complete `GraniteSpeechFeatureExtractor` implementation
- Complete `GraniteSpeechProcessor` implementation
- LoRaAdapter integration for FMS Granite Speech
- E2E test suite with real audio validation
- Generation tests for text and audio inputs

---

## Tasks Completed This Week

### Implementation

| Task | Description | Outcome / Results | 
|------|-------------|-------------------|
| GraniteSpeechFeatureExtractor | Implemented mel-spectrogram extraction with proper padding and normalization | Converts raw audio to (batch, mel_seq_len, 160) features |
| GraniteSpeechProcessor | Implemented text+audio processing with audio token expansion | Handles `<\|audio\|>` token expansion and feature alignment |
| LoRaAdapter Integration | Added PEFT LoRA adapter support with `load_adapter()` and `set_adapter()` methods | Enables LoRA adapter loading/toggling for audio inputs |
| Audio Token Handling | Implemented `_get_num_audio_features()` for calculating projected audio token counts | Correctly computes audio tokens after conformer+projector downsampling |
| Test Bug Fixes | Fixed multiple test bugs in granite_speech tests | All tests now passing without xfails |

### Tests

| Task | Description | Outcome / Results |
|------|-------------|-------------------|
| E2E Real Audio Tests | Added `test_fms_e2e_with_real_audio` using LibriSpeech dataset | Validates full pipeline with real audio samples |
| E2E Batch Audio Tests | Added `test_fms_e2e_with_real_audio_batch` for multiple audio samples | Tests batched processing with different audio lengths |
| Generation Tests | Fixed generation tests with proper use_cache handling | Validates greedy and sampling generation modes | 
| Remove xfails | Removed all pytest.mark.xfail decorators as implementations are complete | All 50+ tests now passing |

**Test Migration Report Summary**

| Category               | Coverage | Status                           |
  |------------------------|----------|----------------------------------|
| Model Unit Tests       | 100%     | Complete                         |
| Processor Unit Tests   | 89%      | Missing save/load only           |
| FeatureExtractor Tests | 12 tests | FMS-specific                     |
| Generation Tests       | ~55%     | Core cases covered               |
| LoRA Adapter Tests     | 100%     | Fully implemented (13 tests)     |
| E2E Component Tests    | 6 tests  | Shape validation with real audio |
| HF Equivalence         | 1 test   | Numerical comparison             |

Remaining Gaps:
- No pretrained model integration test (exact transcription validation)
- No processor save/load test
- Beam search not supported (by design)

---

## Plans for Next Week

| Planned Task | Expected Outcome | Assigned To |
|--------------|------------------|-------------|
| Double check minor details | Verify configuration default values, calculation logics, and edge cases | Team |
| Full E2E test using GPU | Run complete pipeline on GPU hardware to validate CUDA compatibility | Team |
| Final code review | Comprehensive code review for quality, documentation, and FMS style conformance | Team |
| (Optional) LoRaAdapter tests | Add unit tests for LoRA adapter loading and toggling functionality | Team |

---

## Challenges / Blockers

| Issue | Description | Impact | Proposed Solution / Support Needed |
|-------|-------------|--------|-----------------------------------|
| No blockers | All major implementation work completed | N/A | N/A |

---

## Individual Contributions (This Week)

| Student Name | Key Contributions | Hours Contributed | Notes |
|--------------|------------------|-------------------|-------|
| Geonsik | Implemented GraniteSpeechFeatureExtractor and GraniteSpeechProcessor, fixed generation tests | 12 hours | Core audio processing pipeline complete |
| In Keun | Added LoRaAdapter feature, created E2E real audio tests, fixed test bugs | 10 hours | LoRA integration ready for fine-tuning workflows |
| Aneesh | Implemented audio token calculation logic, added batch E2E tests | 9 hours | Proper audio-to-token mapping validated |
| Zach | Fixed multiple test bugs, removed xfails, validated test suite | 8 hours | All tests now passing |

---

## Feedback / Requests from Supervisors

**Specific Questions / Feedback Needed:**

- Final code review session? 


---

## Appendix


### Key Implementation Details

**GraniteSpeechFeatureExtractor:**
- Converts raw audio waveforms to mel-spectrogram features
- Output shape: `(batch, mel_seq_len, 160)` where 160 = n_mels * 2
- Handles variable-length audio with proper padding and masking

**GraniteSpeechProcessor:**
- Combines feature extraction with tokenization
- Expands `<|audio|>` placeholder tokens to match projected audio feature count
- Calculates audio embed sizes for model forward pass

**LoRaAdapter Integration:**
- Optional PEFT dependency for LoRA adapter support
- `load_adapter(path)` - Load LoRA weights onto decoder
- `set_adapter(adapter_name)` - Toggle adapter on/off
- `has_lora_adapter` config flag for runtime behavior

### Test Coverage Summary

| Test Category | Test Count | Status  |
|---------------|------------|---------|
| Component Tests | 20+ | Passing |
| E2E Tests | 5+ | Skipped |
| Generation Tests | 10+ | Passing |
| Integration Tests | 15+ | Passing |

