# HPML Project: Granite Speech Integration in Foundation Model Stack (FMS)

## Team Information

- **Team Name**: Columbia-HPML-Granite
- **Faculty Supervisors**:
  - Dr. Kaoutar El Maghraoui (IBM Research)
  - Dr. Rashed Bhatti (IBM Research)
- **Student Team Members**:
  - Aneesh Durai ([GitHub](https://github.com/aneeshdurai))
  - Geonsik Moon ([GitHub](https://github.com/gsmoon97))
  - In Keun Kim ([GitHub](https://github.com/nearKim))
  - Zachary Zusin ([GitHub](https://github.com/zacharyzusin))

---

## 1. Problem Statement

Integrate IBM's Granite Speech 3.3 8B model into the Foundation Model Stack (FMS) to enable end-to-end execution under `torch.compile`, evaluate its performance against eager execution, and document what makes a speech model compile-efficient inside FMS.

**Key Challenges:**

- Implement three core components from scratch: Conformer encoder, Q-Former projector, and multimodal integration
- Achieve numerical equivalence with HuggingFace reference implementation
- Design compile-friendly architecture adhering to FMS conventions
- Handle HuggingFace-to-FMS weight conversion for seamless checkpoint loading
- Support audio preprocessing pipeline (feature extraction and tokenization)
- Integrate LoRA adapters for efficient fine-tuning

**Research Questions:**

1. Can a multimodal speech-to-text model achieve full compatibility with `torch.compile`?
2. What architectural patterns make speech models compile-efficient?
3. How does encoder vs. decoder latency distribution affect optimization strategy?

---

## 2. Model Description

### Architecture Overview

Granite Speech is a multimodal speech-to-text model combining three main components:

**1. Conformer Encoder (Acoustic Feature Extraction)**

- **Architecture**: 10 conformer blocks with Macaron-style feedforward
- **Components**: Self-attention + Convolution + Feedforward (pre/post attention)
- **Key Features**:
  - Shaw's relative positional embeddings (better for variable-length audio)
  - Depthwise-separable convolution with GLU activation
  - Half-step residual scaling (0.5× for feedforward, 1.0× for attention/conv)
  - Mid-layer CTC supervision at block 8 for auxiliary loss
- **Input**: Mel-spectrogram features (batch, seq_len, 160)
- **Output**: Acoustic embeddings (batch, seq_len, 1024)

**2. Q-Former Projector (Audio-to-Text Bridge)**

- **Architecture**: BLIP-2 inspired query-based transformer
- **Components**: 6 Q-Former layers (self-attention → cross-attention → feedforward)
- **Key Features**:
  - Learnable query embeddings (32 queries, initialized from N(0,1))
  - Window-based processing for temporal downsampling (15× compression)
  - Cross-attention to encoder hidden states
- **Input**: Acoustic embeddings (batch, T, 1024)
- **Output**: Projected audio tokens (batch, ~T/15, 2048)

**3. Granite Decoder (Language Model)**

- **Architecture**: Granite 3.3 8B decoder-only transformer
- **Components**: 40 transformer blocks with GQA and SwiGLU
- **Key Features**:
  - LoRA adapters (rank-64 on query/value projections)
  - 128k token context length
  - Multimodal attention over merged audio-text embeddings
- **Input**: Merged embeddings (batch, audio_tokens + text_tokens, 2048)
- **Output**: Text logits (batch, seq_len, 49160)

### Framework and Implementation

- **Framework**: PyTorch 2.5.1+
- **FMS Integration**: Custom components following FMS ModelConfig patterns
- **Custom Implementations**:
  - `fms/models/conformer.py` (385 lines) - Conformer encoder with CTC
  - `fms/modules/projector.py` (290 lines) - Q-Former projector
  - `fms/models/granite_speech.py` (860 lines) - Main multimodal model
  - `fms/models/granite_speech.py` - GraniteSpeechFeatureExtractor (mel-spectrogram extraction)
  - `fms/models/granite_speech.py` - GraniteSpeechProcessor (audio token expansion)

### Key Technical Decisions

1. **Zero HuggingFace Dependencies**: Pure PyTorch/torchaudio implementation for feature extraction
2. **Placeholder Mechanism**: Prevents tokenizer issues with repeated `<|audio|>` tokens
3. **Window-Based Downsampling**: 15× temporal compression in projector
4. **Serialization Adapters**: Complete HF→FMS weight conversion pipeline
5. **Component-Level Commenting**: ~62 minimalistic comments added for readability

---

## 3. Final Results Summary

### Performance Metrics (from Week 2 Profiling)

**Hardware**: NVIDIA H200 GPU
**Model**: Granite Speech 3.3 8B (HuggingFace baseline)
**Precision**: bf16

| Metric                     | 3s Audio | 10s Audio | 30s Audio |
| -------------------------- | -------- | --------- | --------- |
| **Total Latency (ms)**     | 489.4    | 532.3     | 2662.2    |
| **Encoder Latency (ms)**   | 17.7     | 18.2      | 18.8      |
| **Decoder Latency (ms)**   | 471.7    | 514.1     | 2643.4    |
| **Encoder % of Total**     | 3.6%     | 3.4%      | 0.7%      |
| **Real-Time Factor (RTF)** | 0.163    | 0.053     | 0.089     |
| **Peak Memory (MB)**       | 17,353   | 17,376    | 17,430    |
| **Throughput (tokens/s)**  | 120.7    | 249.0     | 162.1     |

**Key Finding**: Granite Speech is **decoder-bound** (96.4-99.3% of latency in decoder), encoder optimization has minimal impact on end-to-end performance.

### Implementation Completeness

| Component                  | Status   | Test Files                               | Test Count |
| -------------------------- | -------- | ---------------------------------------- | ---------- |
| **Conformer Encoder**      | Complete | test_conformer.py, test_conformer_hf.py  | 28 + 25    |
| **Q-Former Projector**     | Complete | test_projector.py                        | 20         |
| **Granite Speech Model**   | Complete | test_granite_speech.py                   | 62         |
| **Generation Support**     | Complete | test_granite_speech_generation.py        | 11         |
| **Performance Benchmarks** | Complete | test_conformer_simple.py                 | 3          |
| **HF Equivalence**         | Complete | hf_equivalence/test_granite_speech.py    | 6          |
| **Activation Parity**      | Complete | test_granite_speech_activation_parity.py | 1          |
| **torch.compile Parity**   | Complete | test_granite_speech_torch_compile.py     | 1          |

**Total Test Count**: 157 tests across 9 test files
**Code Volume**: ~2,535 lines of production code, ~1,500+ lines of tests

### Compression Pipeline

```
Raw Audio (16kHz) → Mel Features → Encoder → Projector → Decoder
    16000 samples      100 frames   100 embeds   ~7 tokens   text
         ↓ 160×           ↓ 1×         ↓ 15×       ↓ generation
```

**Overall Compression**: 16000 samples → ~7 tokens ≈ **2285× compression**

---

## 4. Reproducibility Instructions

### A. Requirements

**System Requirements:**

- Python 3.11 or 3.12
- CUDA-capable GPU (recommended: NVIDIA H200 or A100)
- 32GB+ RAM
- ~50GB disk space for model checkpoints

**Install Dependencies:**

```bash
cd foundation-model-stack

# Install FMS with development dependencies
pip install -e ".[dev]"

# Additional dependencies for audio processing
pip install torchaudio librosa soundfile
```

**Dependency List** (from `pyproject.toml`):

- Core: `torch >= 2.5.1`
- Dev: `pytest==8.3.4`, `transformers==4.55.4`, `peft==0.14.0`, `lm_eval==0.4.7`
- Audio: `torchaudio`, `librosa`, `soundfile`

---

### B. Wandb Dashboard

N/A - This project did not use Weights & Biases for experiment tracking.

---

### C. Training and Inference

**Training:**
N/A - This project focused on integration and equivalence testing, not training from scratch. The model uses pretrained Granite Speech checkpoints from HuggingFace.

**Inference:**

To run end-to-end inference with the FMS implementation (see [`notebooks/granite_speech_inference.ipynb`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/notebooks/granite_speech_inference.ipynb) for full example):

```python
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download

from fms.models import get_model
from fms.models.granite_speech import GraniteSpeechFeatureExtractor, GraniteSpeechProcessor
from fms.utils.generation import generate
from fms.utils.tokenizers import get_tokenizer

# 1. Load model using FMS get_model API
model_id = "ibm-granite/granite-speech-3.3-8b"
model_path = snapshot_download(model_id)

model = get_model(
    "granite_speech",
    "3.3-8b",
    model_path=model_path,
    source="hf",
    device_type="cuda",
    data_type=torch.bfloat16,
)
model.eval()

# 2. Get tokenizer
tokenizer_wrapper = get_tokenizer(model_id)
tokenizer = getattr(tokenizer_wrapper, 'tokenizer', tokenizer_wrapper)

# 3. Load audio from LibriSpeech dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]
audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)

# 4. Build chat-formatted prompt
chat = [
    {"role": "system", "content": "You are Granite, developed by IBM. You are a helpful AI assistant"},
    {"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"}
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# 5. Process inputs (feature extraction + tokenization)
processor = GraniteSpeechProcessor(GraniteSpeechFeatureExtractor(), tokenizer)
inputs = processor(text=[prompt], audio=audio, return_tensors="pt")
inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

# 6. Generate transcription
with torch.no_grad():
    output_ids = generate(
        model,
        inputs["input_ids"],
        max_new_tokens=200,
        do_sample=False,
        use_cache=True,
        extra_kwargs={
            "input_features": inputs["input_features"],
            "input_features_mask": inputs.get("input_features_mask"),
            "attention_mask": inputs.get("attention_mask"),
        },
    )

# 7. Decode output
input_length = inputs["input_ids"].shape[1]
new_tokens = output_ids[:, input_length:]
transcription = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

print(f"Ground Truth:  {sample['text']}")
print(f"Transcription: {transcription.upper()}")
```

**Key Features:**

- Uses FMS `get_model()` API for model loading with HuggingFace checkpoint conversion
- Automatic feature extraction from raw audio to mel-spectrograms
- Chat template formatting for proper instruction following
- Audio token placeholder expansion handled by `GraniteSpeechProcessor`
- Efficient generation with KV caching support

---

### D. Evaluation

To evaluate the FMS implementation:

```bash
# Run all Granite Speech tests (CPU tests)
pytest tests/models/test_granite_speech.py \
       tests/models/test_granite_speech_generation.py \
       tests/models/test_conformer.py \
       tests/models/test_conformer_hf.py \
       tests/models/test_projector.py -v

# Run performance benchmarks (CPU tests)
pytest tests/models/hf_equivalence/test_conformer_simple.py -v

# Run HuggingFace numerical equivalence tests (requires GPU)
pytest tests/models/hf_equivalence/test_granite_speech.py -v

# Run torch.compile parity test (requires GPU)
pytest tests/models/hf_equivalence/test_granite_speech_torch_compile.py -v

# Run activation parity test for component-level debugging (requires GPU)
pytest tests/models/hf_equivalence/test_granite_speech_activation_parity.py -v

# Run all tests at once (157 tests)
pytest tests/models/test_granite_speech*.py \
       tests/models/test_conformer*.py \
       tests/models/test_projector.py \
       tests/models/hf_equivalence/test_granite_speech*.py \
       tests/models/hf_equivalence/test_conformer_simple.py -v
```

**Test Categories:**

- **Unit Tests**: Component-level validation (shapes, gradients, numerical stability)
- **Integration Tests**: End-to-end forward pass, weight conversion, LoRA loading
- **Equivalence Tests**: Numerical parity with HuggingFace (logits, activations)
- **Performance Tests**: Baseline CPU benchmarks, compile speedup validation

---

### E. Quickstart: Minimum Reproducible Result

To reproduce the minimum reported result (passing all CPU tests):

```bash
# Step 1: Clone and setup repository
git clone https://github.com/columbia-hpml-granite/foundation-model-stack.git
cd foundation-model-stack
git checkout granite-speech

# Step 2: Install dependencies
pip install -e ".[dev]"
pip install datasets soundfile torchaudio huggingface_hub peft

# Step 3: Run core test suite (no GPU required, ~2-3 minutes)
pytest tests/models/test_granite_speech.py \
       tests/models/test_granite_speech_generation.py \
       tests/models/test_conformer.py \
       tests/models/test_conformer_hf.py \
       tests/models/test_projector.py \
       -v --tb=short

# Expected output: 146 tests passed (CPU tests only)
```

**For GPU-based Equivalence Testing:**

```bash
# Download HuggingFace checkpoint (required for equivalence tests)
# Model will auto-download on first run (~32GB)

# Run all GPU-required tests (requires CUDA GPU, ~10-15 minutes)
pytest tests/models/hf_equivalence/test_granite_speech.py \
       tests/models/hf_equivalence/test_granite_speech_torch_compile.py \
       tests/models/hf_equivalence/test_granite_speech_activation_parity.py \
       -v -s

# Expected output: 8 additional tests (numerical equivalence, compile parity, activation debugging)
```

**For Complete Inference Demo:**

```bash
# Run the inference notebook (requires GPU, downloads model checkpoint)
jupyter notebook notebooks/granite_speech_inference.ipynb

# Or run as Python script
python -c "
import torch
from datasets import load_dataset
from fms.models import get_model
from fms.models.granite_speech import GraniteSpeechProcessor, GraniteSpeechFeatureExtractor
from fms.utils.generation import generate
from fms.utils.tokenizers import get_tokenizer

# Load sample audio
dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
audio = torch.tensor(dataset[0]['audio']['array'])

# Load model
model = get_model('granite_speech', '3.3-8b', source='hf', device_type='cuda')

# Run inference
tokenizer = get_tokenizer('ibm-granite/granite-speech-3.3-8b').tokenizer
processor = GraniteSpeechProcessor(GraniteSpeechFeatureExtractor(), tokenizer)
inputs = processor(text=['<|audio|>Transcribe:'], audio=audio, return_tensors='pt')
outputs = generate(model, inputs['input_ids'].cuda(), max_new_tokens=50,
                   extra_kwargs={'input_features': inputs['input_features'].cuda()})
print(tokenizer.decode(outputs[0]))
"
```

---

## 5. Notes

### Working Inference Example

A complete, executable inference notebook is provided at:
**[`notebooks/granite_speech_inference.ipynb`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/notebooks/granite_speech_inference.ipynb)**

This notebook demonstrates:

- Model loading with FMS `get_model()` API
- Audio preprocessing with `GraniteSpeechFeatureExtractor`
- Text processing with chat template formatting
- End-to-end transcription generation with real LibriSpeech audio samples
- Testing both 3.3-8B and 3.3-2B model variants

### Repository Structure

```
foundation-model-stack/
├── fms/
│   ├── models/
│   │   ├── conformer.py          # Conformer encoder (385 lines)
│   │   ├── granite_speech.py     # Main model + FeatureExtractor + Processor (860 lines)
│   │   └── hf/
│   │       └── utils.py          # HF config mapping
│   └── modules/
│       └── projector.py          # Q-Former projector (290 lines)
├── notebooks/
│   └── granite_speech_inference.ipynb       # Complete inference demo
├── tests/
│   ├── models/
│   │   ├── test_granite_speech.py           # End-to-end tests (62 tests)
│   │   ├── test_granite_speech_generation.py # Generation tests (11 tests)
│   │   ├── test_conformer.py                # Conformer unit tests (28 tests)
│   │   ├── test_conformer_hf.py             # HF-aligned encoder tests (25 tests)
│   │   ├── test_projector.py                # Projector tests (20 tests)
│   │   └── hf_equivalence/
│   │       ├── test_conformer_simple.py              # Performance benchmarks (3 tests)
│   │       ├── test_granite_speech.py                # Numerical equivalence (6 tests)
│   │       ├── test_granite_speech_torch_compile.py  # Compile parity (1 test)
│   │       └── test_granite_speech_activation_parity.py # Activation debugging (1 test)
└── pyproject.toml                           # Dependencies and build config
```

### Key Implementation Files

All implementations are on the **`granite-speech`** branch:

- Conformer: [`fms/models/conformer.py`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/fms/models/conformer.py)
- Projector: [`fms/modules/projector.py`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/fms/modules/projector.py)
- Granite Speech: [`fms/models/granite_speech.py`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/fms/models/granite_speech.py)
- Tests: [`tests/models/test_granite_speech.py`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/tests/models/test_granite_speech.py)

### Documentation

Technical documentation available in the repository:

- [`ARCHITECTURE_FLOW.md`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/ARCHITECTURE_FLOW.md) - End-to-end architecture and data flow
- [`CONFORMER_IMPLEMENTATION.md`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/CONFORMER_IMPLEMENTATION.md) - Conformer design decisions
- [`PROJECTOR_IMPLEMENTATION.md`](https://github.com/columbia-hpml-granite/foundation-model-stack/blob/granite-speech/PROJECTOR_IMPLEMENTATION.md) - Q-Former implementation details
- [Weekly Reports](https://github.com/columbia-hpml-granite/granite-docs/tree/main/docs/Weekly) - Week-by-week progress logs

### Test Execution Notes

**CPU Tests** (146 tests, ~2-3 minutes):

- All component unit tests (conformer, projector, granite_speech)
- Shape validation, gradient flow, numerical stability
- Configuration tests and serialization
- Can run on any machine without GPU

**GPU Tests** (11 tests, ~10-15 minutes):

- Numerical equivalence with HuggingFace (logits comparison)
- torch.compile parity testing
- Activation-level debugging with module hooks
- Requires CUDA GPU and ~32GB model checkpoint download

**Test Organization**:

- Unit tests in `tests/models/` - Component-level validation
- Performance tests in `tests/models/hf_equivalence/` - Benchmarking and profiling
- Equivalence tests in `tests/models/hf_equivalence/` - HF numerical parity

### Contact Information

**Project Repository**: [columbia-hpml-granite/foundation-model-stack](https://github.com/columbia-hpml-granite/foundation-model-stack/tree/granite-speech)
**Documentation Site**: [columbia-hpml-granite/granite-docs](https://github.com/columbia-hpml-granite/granite-docs)

For questions or issues related to this implementation, please contact the team members via GitHub.

### References

- **Granite Speech Paper**: [arXiv:2505.08699](https://arxiv.org/pdf/2505.08699)
- **HuggingFace Model**: [ibm-granite/granite-speech-3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- **IBM Granite Documentation**: [IBM Granite Models](https://www.ibm.com/granite/docs/models/speech)
- **Conformer Paper**: [arXiv:2005.08100](https://arxiv.org/abs/2005.08100)
- **BLIP-2 Paper** (Q-Former inspiration): [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- **Shaw's Relative Positional Embeddings**: [arXiv:1803.02155](https://arxiv.org/abs/1803.02155)

### Acknowledgments

This project was conducted as part of Columbia University's High Performance Machine Learning course in collaboration with IBM Research. Special thanks to Dr. Kaoutar El Maghraoui and Dr. Rashed Bhatti for their guidance and support throughout the project.

---

_Final Report Generated: December 19, 2024_
