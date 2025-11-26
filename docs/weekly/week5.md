---
title: Week 5
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

**Project Objective:** Integrate IBM's Granite Speech 3.3 8B model into the Foundation Model Stack to enable end-to-end execution under `torch.compile`, focusing on Speech Projector implementation and comprehensive testing infrastructure.

---

## Overall Progress Summary

**% Completion:** 75%

**Key Milestones Achieved This Week:**
- ✅ Complete Speech Projector Q-Former implementation (633 lines)
- ✅ CTC mid-layer supervision added to Conformer Encoder
- ✅ Full Granite Speech model integration (encoder + projector + decoder)
- ✅ HuggingFace → FMS weight conversion adapters
- ✅ Comprehensive HF-aligned test suites (1,321 lines of tests)
- ✅ All 22 end-to-end integration tests passing

**Deliverables Submitted:**
- Complete Speech Projector module (`fms/modules/projector.py`)
- Enhanced Conformer Encoder with CTC (`fms/models/conformer.py`)
- Granite Speech model integration (`fms/models/granite_speech.py` )
- HF-aligned test suites

---

## Tasks Completed This Week

| Task | Description | Outcome / Results | Responsible Member |
|------|-------------|-------------------|-------------------|
| **Complete Projector Implementation** | Implemented all Q-Former components: QFormerSelfAttention, QFormerCrossAttention, QFormerFeedForward, QFormerLayer | 633 lines, fully functional with learnable queries, multi-head attention, and feed-forward networks | Geonsik Moon |
| **CTC Mid-Layer Supervision** | Added CTC output layers and mid-layer feedback to ConformerEncoder | `use_ctc` config option, `out`/`out_mid` linear layers, softmax feedback at layer `num_layers // 2` | Geonsik Moon |
| **Granite Speech Model Integration** | Connected Conformer encoder → Q-Former projector → Granite decoder into unified forward path | `get_audio_features()`, `get_merged_audio_embeddings()`, multimodal `forward()` with labels support | In Keun Kim |
| **HF Weight Conversion** | Implemented HuggingFace → FMS weight name mapping and K-V weight splitting | `_hf_to_fms_names()`, `_split_kv_weights()`, registered as FMS serialization adapter | In Keun Kim |
| **HF-Aligned Projector Tests** | Created comprehensive test suite following HuggingFace patterns | 401 lines, 21 tests covering window processing, query initialization, gradient flow | In Keun Kim |
| **HF-Aligned Encoder Tests** | Created CTC-focused test suite for Conformer | 434 lines, 25 tests covering CTC configuration, mid-layer logic, attention distances | In Keun Kim |
| **End-to-End Model Tests** | Created integration tests for complete Granite Speech model | 486 lines, 22 tests covering forward pass, gradient flow, weight conversion, stability | In Keun Kim |

---

## Technical Achievements This Week

### 1. Complete Q-Former Projector (`fms/modules/projector.py`)

**Components Implemented:**
```python
class SpeechProjectorConfig(ModelConfig):
    encoder_dim: int = 1024      # Conformer output
    decoder_dim: int = 2048      # Granite input
    num_queries: int = 32        # Learnable queries
    num_hidden_layers: int = 6   # Q-Former layers
    num_attention_heads: int = 8
    intermediate_size: int = 4096

class QFormerSelfAttention(nn.Module):   # Multi-head self-attention on queries
class QFormerCrossAttention(nn.Module):  # Cross-attention: queries → encoder
class QFormerFeedForward(nn.Module):     # FFN with residual + LayerNorm
class QFormerLayer(nn.Module):           # Self-attn → Cross-attn → FFN
class SpeechProjector(nn.Module):        # Full projector with output projection
```

**Architecture Flow:**
```
Conformer output (batch, audio_len, 1024)
    ↓
Learnable Queries (1, 32, 1024) expanded to batch
    ↓
Q-Former Layer 1: Self-attn → Cross-attn → FFN
    ↓
Q-Former Layer 2: Self-attn → Cross-attn → FFN
    ↓
LayerNorm + Output Projection
    ↓
Projected output (batch, 32, 4096)
```

### 2. CTC Mid-Layer Supervision (`fms/models/conformer.py`)

**Added to ConformerConfig:**
```python
output_dim: int = 42      # CTC vocabulary size
use_ctc: bool = True      # Enable mid-layer CTC
```

**Forward Pass Enhancement:**
```python
mid_layer = len(self.blocks) // 2
for idx, block in enumerate(self.blocks, start=1):
    x = block(x, attention_dists)

    # Mid-layer CTC feedback (HF-aligned)
    if self.config.use_ctc and idx == mid_layer:
        x_mid = self.out(x)  # (batch, seq_len, output_dim)
        x = x + self.out_mid(F.softmax(x_mid, dim=-1))
```

### 3. Granite Speech Integration (`fms/models/granite_speech.py`)

**Key Methods:**
```python
def get_audio_features(self, input_features):
    """Encode audio → Project to decoder space."""
    encoder_embeds = self.encoder(input_features)
    projected_embeds = self.projector(encoder_embeds)
    return projected_embeds

def get_merged_audio_embeddings(self, input_ids, audio_features):
    """Replace audio token placeholders with projected features."""
    is_audio_index = input_ids == self.audio_token_index
    llm_input_ids = torch.where(is_audio_index, 0, input_ids)
    inputs_embeds = self.get_input_embeddings()(llm_input_ids)
    inputs_embeds = inputs_embeds.masked_scatter(
        is_audio_index.unsqueeze(-1), audio_features
    )
    return inputs_embeds

def forward(self, input_ids, input_features, labels=None, ...):
    """Full multimodal forward pass."""
    if input_features is not None:
        audio_embeds = self.get_audio_features(input_features)
        inputs_embeds = self.get_merged_audio_embeddings(input_ids, audio_embeds)
    else:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    decoder_out, cache = self.decoder(inputs_embeds, ...)
    logits = self.lm_head(decoder_out)

    if labels is not None:
        loss = cross_entropy(logits, labels)
    return logits, loss
```

### 4. HuggingFace Weight Conversion

**Registered Adapter Steps:**
```python
serialization.register_adapter(
    "granite_speech",
    "hf",
    ["hf_to_fms_names", "split_kv_weights", "weight_fusion"],
)
```

**Key Mappings:**
| HuggingFace | FMS |
|-------------|-----|
| `encoder.input_linear` | `encoder.input_proj` |
| `encoder.layers.{i}.ff1.pre_norm` | `encoder.blocks.{i}.ff1.norm` |
| `encoder.layers.{i}.attn.to_kv` | `encoder.blocks.{i}.attn.to_k` + `to_v` (split) |
| `projector.query` | `projector.query_embeds` |
| `projector.qformer.encoder.layer.{i}` | `projector.layers.{i}` |
| `language_model.model.layers` | `decoder.layers` |

---

## Architecture Diagrams (Recap)

### Granite Speech End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GRANITE SPEECH 3.3 8B ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────────┘

   Audio Input                    Text Input
   (Log-Mel Spectrogram)          (Token IDs)
        │                              │
        ▼                              ▼
┌───────────────────┐          ┌───────────────────┐
│   Input Features  │          │   Token Embedding │
│ (B, T, 160)       │          │   (B, S, 4096)    │
└─────────┬─────────┘          └─────────┬─────────┘
          │                              │
          ▼                              │
┌───────────────────────────────┐        │
│      CONFORMER ENCODER        │        │
│  ┌─────────────────────────┐  │        │
│  │    Input Projection     │  │        │
│  │    (160 → 1024)         │  │        │
│  └───────────┬─────────────┘  │        │
│              ▼                │        │
│  ┌─────────────────────────┐  │        │
│  │   Conformer Block ×8    │  │        │
│  │  ┌───────────────────┐  │  │        │
│  │  │ FFN₁ (half-step)  │  │  │        │
│  │  │ Multi-Head Attn   │  │  │        │
│  │  │ Conv Module       │  │  │        │
│  │  │ FFN₂ (half-step)  │  │  │        │
│  │  │ LayerNorm         │  │  │        │
│  │  └───────────────────┘  │  │        │
│  └───────────┬─────────────┘  │        │
│              │ ←── CTC at     │        │
│              │     layer 8    │        │
│  ┌───────────┴─────────────┐  │        │
│  │   Conformer Block ×8    │  │        │
│  └───────────┬─────────────┘  │        │
│              ▼                │        │
│     (B, T, 1024)              │        │
└───────────────┬───────────────┘        │
                │                        │
                ▼                        │
┌───────────────────────────────┐        │
│      Q-FORMER PROJECTOR       │        │
│  ┌─────────────────────────┐  │        │
│  │  Learnable Queries      │  │        │
│  │  (1, 32, 1024)          │  │        │
│  └───────────┬─────────────┘  │        │
│              ▼                │        │
│  ┌─────────────────────────┐  │        │
│  │   Q-Former Layer ×2     │  │        │
│  │  ┌───────────────────┐  │  │        │
│  │  │ Self-Attention    │──┼──┼────────┤
│  │  │ Cross-Attention   │◄─┼──┘ encoder│
│  │  │ Feed-Forward      │  │  │ hidden │
│  │  │ + Residual + LN   │  │  │ states │
│  │  └───────────────────┘  │  │        │
│  └───────────┬─────────────┘  │        │
│              ▼                │        │
│  ┌─────────────────────────┐  │        │
│  │   Output Projection     │  │        │
│  │   (1024 → 4096)         │  │        │
│  └───────────┬─────────────┘  │        │
│              ▼                │        │
│     (B, 32, 4096)             │        │
└───────────────┬───────────────┘        │
                │                        │
                ▼                        ▼
        ┌───────────────────────────────────────┐
        │         EMBEDDING MERGE               │
        │  ┌─────────────────────────────────┐  │
        │  │ <audio> tokens → Audio Embeds   │  │
        │  │ <text> tokens  → Text Embeds    │  │
        │  └─────────────────────────────────┘  │
        │         (B, 32+S, 4096)               │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │         GRANITE DECODER               │
        │  ┌─────────────────────────────────┐  │
        │  │   Transformer Block ×40         │  │
        │  │  • RMSNorm                       │  │
        │  │  • Multi-Head Attention (GQA)   │  │
        │  │  • RMSNorm                       │  │
        │  │  • SwiGLU FFN                    │  │
        │  └─────────────────────────────────┘  │
        │         (B, 32+S, 4096)               │
        └───────────────────┬───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │            LM HEAD                    │
        │         (4096 → 49160)                │
        └───────────────────┬───────────────────┘
                            │
                            ▼
                    Output Logits
                   (B, 32+S, 49160)
```

### Q-Former Projector Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Q-FORMER PROJECTOR ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Encoder Hidden States                 Learnable Queries
    (B, T, 1024)                          (1, 32, 1024)
         │                                      │
         │                                      │ expand to batch
         │                                      ▼
         │                               (B, 32, 1024)
         │                                      │
         │    ┌─────────────────────────────────┼─────────────────────────────┐
         │    │              Q-FORMER LAYER (×2)                              │
         │    │                                 │                             │
         │    │                                 ▼                             │
         │    │    ┌────────────────────────────────────────────────┐        │
         │    │    │            SELF-ATTENTION                      │        │
         │    │    │  ┌──────────────────────────────────────────┐  │        │
         │    │    │  │  Q = W_q · queries    (B, 32, 1024)      │  │        │
         │    │    │  │  K = W_k · queries    (B, 32, 1024)      │  │        │
         │    │    │  │  V = W_v · queries    (B, 32, 1024)      │  │        │
         │    │    │  │                                          │  │        │
         │    │    │  │  Attn = softmax(QK^T / √d) · V           │  │        │
         │    │    │  │  Output = W_o · Attn + queries (residual)│  │        │
         │    │    │  └──────────────────────────────────────────┘  │        │
         │    │    │  LayerNorm                                     │        │
         │    │    └────────────────────────┬───────────────────────┘        │
         │    │                             │                                │
         │    │                             ▼                                │
         │    │    ┌────────────────────────────────────────────────┐        │
         │    │    │           CROSS-ATTENTION                      │        │
         │    │    │  ┌──────────────────────────────────────────┐  │        │
         └────┼────┼──│► K = W_k · encoder    (B, T, 1024)       │  │        │
              │    │  │  V = W_v · encoder    (B, T, 1024)       │  │        │
              │    │  │  Q = W_q · queries    (B, 32, 1024)      │  │        │
              │    │  │                                          │  │        │
              │    │  │  Attn = softmax(QK^T / √d) · V           │  │        │
              │    │  │  Output = W_o · Attn + queries (residual)│  │        │
              │    │  └──────────────────────────────────────────┘  │        │
              │    │  LayerNorm                                     │        │
              │    └────────────────────────┬───────────────────────┘        │
              │                             │                                │
              │                             ▼                                │
              │    ┌────────────────────────────────────────────────┐        │
              │    │           FEED-FORWARD                         │        │
              │    │  ┌──────────────────────────────────────────┐  │        │
              │    │  │  h = GELU(W_1 · x)     (B, 32, 4096)     │  │        │
              │    │  │  out = W_2 · h + x     (B, 32, 1024)     │  │        │
              │    │  └──────────────────────────────────────────┘  │        │
              │    │  LayerNorm                                     │        │
              │    └────────────────────────┬───────────────────────┘        │
              │                             │                                │
              └─────────────────────────────┼────────────────────────────────┘
                                            │
                                            ▼
                               ┌────────────────────────┐
                               │   Final LayerNorm      │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │   Output Projection    │
                               │   (1024 → 4096)        │
                               └───────────┬────────────┘
                                           │
                                           ▼
                                    (B, 32, 4096)
                               Projected Audio Features
```

### Tensor Shape Flow Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TENSOR SHAPE TRANSFORMATIONS                        │
└────────────────────────────────────────────────────────────────────────────┘

Component                    Input Shape              Output Shape
─────────────────────────────────────────────────────────────────────────────
Audio Features               (B, T, 160)              -
Conformer Input Proj         (B, T, 160)              (B, T, 1024)
Conformer Blocks             (B, T, 1024)             (B, T, 1024)
CTC out layer                (B, T, 1024)             (B, T, 42)
CTC out_mid layer            (B, T, 42)               (B, T, 1024)
─────────────────────────────────────────────────────────────────────────────
Query Embeddings             (1, 32, 1024)            (B, 32, 1024)
Q-Former Self-Attn           (B, 32, 1024)            (B, 32, 1024)
Q-Former Cross-Attn          Q:(B,32,1024) K,V:(B,T,1024)  (B, 32, 1024)
Q-Former FFN                 (B, 32, 1024)            (B, 32, 1024)
Projector Output Proj        (B, 32, 1024)            (B, 32, 4096)
─────────────────────────────────────────────────────────────────────────────
Text Token Embedding         (B, S)                   (B, S, 4096)
Merged Embeddings            Audio + Text             (B, 32+S, 4096)
Granite Decoder              (B, 32+S, 4096)          (B, 32+S, 4096)
LM Head                      (B, 32+S, 4096)          (B, 32+S, 49160)
─────────────────────────────────────────────────────────────────────────────

Legend: B=batch, T=audio_seq_len, S=text_seq_len
        Typical values: T≈500 (10s audio), S≈128 (text tokens)
```

## FMS Code Structure (Recap)

### Module Dependencies

```
fms/
├── models/
│   ├── granite_speech.py      # Main model (imports conformer, projector, granite)
│   │   ├── GraniteSpeechConfig
│   │   ├── GraniteSpeech
│   │   └── HF weight adapters
│   │
│   ├── conformer.py           # Conformer encoder with CTC
│   │   ├── ConformerConfig
│   │   ├── ConformerBlock (FFN1 → Attn → Conv → FFN2 → LN)
│   │   └── ConformerEncoder
│   │
│   └── granite.py             # Granite decoder (existing FMS)
│
└── modules/
    └── projector.py           # Q-Former projector
        ├── SpeechProjectorConfig
        ├── QFormerSelfAttention
        ├── QFormerCrossAttention
        ├── QFormerFeedForward
        ├── QFormerLayer
        └── SpeechProjector
```

### Call Graph

```
GraniteSpeech.forward()
    │
    ├── get_audio_features(input_features)
    │       │
    │       ├── ConformerEncoder.forward()
    │       │       ├── input_proj(x)
    │       │       ├── for block in blocks:
    │       │       │       └── ConformerBlock.forward()
    │       │       │               ├── ff1(x) * 0.5 + x
    │       │       │               ├── attn(x) + x
    │       │       │               ├── conv(x) + x
    │       │       │               ├── ff2(x) * 0.5 + x
    │       │       │               └── post_norm(x)
    │       │       └── CTC feedback at layer N/2
    │       │
    │       └── SpeechProjector.forward()
    │               ├── expand queries to batch
    │               ├── for layer in layers:
    │               │       └── QFormerLayer.forward()
    │               │               ├── self_attention(queries)
    │               │               ├── cross_attention(queries, encoder_states)
    │               │               └── feed_forward(queries)
    │               ├── layer_norm(queries)
    │               └── output_proj(queries)
    │
    ├── get_merged_audio_embeddings(input_ids, audio_features)
    │       ├── Find audio token positions
    │       ├── Get text embeddings
    │       └── masked_scatter audio features
    │
    └── decoder.forward(inputs_embeds) → lm_head(output)
```

---

## Test Execution Results

### Current Status

```bash
$ pytest tests/models/test_granite_speech.py tests/models/test_conformer_hf.py tests/modules/test_projector_hf.py -v

================== 68 passed, 1 skipped in 174.28s (0:02:54) ===================
```

TBD

---

## Challenges / Blockers

N/A

---

## Individual Contributions (This Week)

| Student Name | Key Contributions | Hours Contributed | Notes |
|--------------|-------------------|-------------------|-------|
| **In Keun Kim** | Added GraniteSpeechConfig and SpeechProjectorConfig classes. Created basic configuration tests and model instantiation tests. | 5                 | |
| **Aneesh Durai** | Completed projector.py TODO implementations (QFormerSelfAttention, QFormerCrossAttention forward methods). Created HF to FMS equivalence test suite. | 7                 | |
| **Geonsik Moon** | Performed performance testing and code validity verification. Established base Q-Former projector interface and architecture design. | 6                 | |
| **Zachary Zusin** | Integrated full Granite Speech model forward path. Added CTC mid-layer supervision to ConformerEncoder. Implemented HF→FMS weight conversion adapters. Created HF-aligned encoder and end-to-end tests. | 5                 | |

---

## Feedback / Requests from Supervisors


---

## Appendix

**Branch:** `granite-speech`

**Files Modified/Created:**
| File | Lines | Description |
|------|-------|-------------|
| `fms/models/granite_speech.py` | 598 | Complete Granite Speech model |
| `fms/modules/projector.py` | 633 | Q-Former projector implementation |
| `fms/models/conformer.py` | 708 | Conformer with CTC support |
| `tests/models/test_granite_speech.py` | 486 | End-to-end model tests |
| `tests/models/test_conformer_hf.py` | 434 | HF-aligned encoder tests |
| `tests/modules/test_projector_hf.py` | 401 | HF-aligned projector tests |
