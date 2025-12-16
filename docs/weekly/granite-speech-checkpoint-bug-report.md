---
title: "Bug Report: Orphaned Checkpoint Files in ibm-granite/granite-speech-3.3-2b"
---


# Bug Report: Orphaned Checkpoint Files in `ibm-granite/granite-speech-3.3-2b`

## Summary

The HuggingFace repository `ibm-granite/granite-speech-3.3-2b` contains orphaned checkpoint files that cause model loading failures in FMS and other glob-based loaders.

---

## Problem Description

When loading the model using FMS:

```python
from fms.models import get_model

model = get_model("hf_pretrained", "ibm-granite/granite-speech-3.3-2b")
```

The following error occurs:

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (42) at non-singleton dimension 0
```

---

## Root Cause

### 1. Repository Contains Two Incompatible Checkpoint Sets

The HuggingFace repository contains **both old and new checkpoint files**:

| File Set | Files | Encoder output_dim | Status |
|----------|-------|-------------------|--------|
| Old (3-shard) | `model-0000X-of-00003.safetensors` | **42** | Orphaned |
| New (4-shard) | `model-0000X-of-00004.safetensors` | **256** | Authoritative |

### 2. Index File Only References New Files

The `model.safetensors.index.json` correctly references **only the 4-shard files**:

```json
{
  "weight_map": {
    "encoder.out.weight": "model-00003-of-00004.safetensors",
    ...
  }
}
```

The 3-shard files are **orphaned** and should have been deleted when the new checkpoint was uploaded.

### 3. Shape Conflicts in Overlapping Keys

Three tensor keys have **conflicting shapes** between old and new files:

| Key | Old (3-shard) | New (4-shard) |
|-----|---------------|---------------|
| `encoder.out.bias` | `(42,)` | `(256,)` |
| `encoder.out.weight` | `(42, 1024)` | `(256, 1024)` |
| `encoder.out_mid.weight` | `(1024, 42)` | `(1024, 256)` |

---

## Why FMS Fails

### FMS Loading Behavior

FMS uses glob pattern matching to discover checkpoint files in `fms/utils/serialization.py`:

```python
# Line 396-402
elif source == "hf":
    glob_pattern_list = ["*.safetensors", "*.bin", "*.pt"]

file_list = list(model_path.glob(glob_pattern_possibility))
checkpoints = sorted(file_list)  # Loads ALL matching files
```

### The Problem

1. FMS **ignores** `model.safetensors.index.json`
2. FMS loads **ALL** `.safetensors` files via glob
3. Both old (42-dim) and new (256-dim) weights are loaded
4. When merging state dicts, shape conflicts cause `RuntimeError`

### Comparison with HuggingFace Transformers

The `transformers` library reads `model.safetensors.index.json` and loads **only referenced files**. FMS does not implement this behavior.

---

## Validation Script Output

Running [`validate_hf_checkpoint_redundancy.py`](/validate_hf_checkpoint_redundancy.py) confirms the issue:

```
[1. Files in Repository]
--------------------------------------------------
Old 3-shard files: 3
  - model-00001-of-00003.safetensors
  - model-00002-of-00003.safetensors
  - model-00003-of-00003.safetensors

New 4-shard files: 4
  - model-00001-of-00004.safetensors
  - model-00002-of-00004.safetensors
  - model-00003-of-00004.safetensors
  - model-00004-of-00004.safetensors

[2. Authoritative Index (model.safetensors.index.json)]
--------------------------------------------------
Referenced files: ['model-00001-of-00004.safetensors', ...]

Orphaned files (NOT in index):
  - model-00001-of-00003.safetensors  ← SHOULD BE DELETED
  - model-00002-of-00003.safetensors  ← SHOULD BE DELETED
  - model-00003-of-00003.safetensors  ← SHOULD BE DELETED

[4. Shape Conflicts (Root Cause of Loading Errors)]
--------------------------------------------------
Found 3 conflicting keys:

  encoder.out.weight:
    OLD: (42, 1024)  (model-00003-of-00003.safetensors)
    NEW: (256, 1024)  (model-00003-of-00004.safetensors)
```

---

## Recommended Fixes

### Option 1: IBM Removes Orphaned Files (Preferred)

Delete the orphaned 3-shard files from the HuggingFace repository:

```bash
huggingface-cli repo delete-file ibm-granite/granite-speech-3.3-2b model-00001-of-00003.safetensors
huggingface-cli repo delete-file ibm-granite/granite-speech-3.3-2b model-00002-of-00003.safetensors
huggingface-cli repo delete-file ibm-granite/granite-speech-3.3-2b model-00003-of-00003.safetensors
```

### Option 2: FMS Respects Index File

Modify `fms/utils/serialization.py` to read `model.safetensors.index.json` and load only referenced files, similar to how `transformers` handles sharded checkpoints.

---

## Current Workaround

Until IBM fixes the repository, use `snapshot_download` with `ignore_patterns` to exclude the orphaned files:

```python
from huggingface_hub import snapshot_download
from fms.models import get_model

# Download checkpoint excluding old 3-shard files
model_path = snapshot_download(
    "ibm-granite/granite-speech-3.3-2b",
    ignore_patterns=["*-of-00003.safetensors"],
)

# Load model from filtered checkpoint
model = get_model(
    "granite_speech",
    "3.3-2b",
    model_path=model_path,
    source="hf",
    device_type="cuda",
    data_type=torch.bfloat16,
)
```

This workaround:
1. Downloads only the 4-shard files (output_dim=256)
2. Skips the orphaned 3-shard files (output_dim=42)
3. Avoids the shape mismatch error

---

## Affected Models

| Model | Status |
|-------|--------|
| `ibm-granite/granite-speech-3.3-2b` | **Affected** (orphaned files present) |
| `ibm-granite/granite-speech-3.3-8b` | OK (no orphaned files) |

---

## References

- HuggingFace Repository: https://huggingface.co/ibm-granite/granite-speech-3.3-2b
- FMS Serialization Code: `fms/utils/serialization.py:396-402`
- Validation Script: `scripts/validate_hf_checkpoint_redundancy.py`
