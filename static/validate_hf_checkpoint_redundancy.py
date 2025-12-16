#!/usr/bin/env python3
"""
Validate HuggingFace Checkpoint Redundancy

Detects orphaned checkpoint files in ibm-granite/granite-speech-3.3-2b that cause
shape mismatch errors when loading with FMS or other glob-based loaders.

Problem: The repo contains both old 3-shard files (output_dim=42) and new 4-shard
files (output_dim=256). Only the 4-shard files are referenced by the index.

Usage:
    python scripts/validate_hf_checkpoint_redundancy.py
"""

import json
from dataclasses import dataclass
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open


MODEL_ID = "ibm-granite/granite-speech-3.3-2b"


@dataclass
class TensorInfo:
    """Metadata for a tensor in a checkpoint file."""
    filename: str
    shape: tuple[int, ...]


@dataclass
class ShapeConflict:
    """A key that exists in both old and new files with different shapes."""
    key: str
    old: TensorInfo
    new: TensorInfo


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a subsection header."""
    print(f"\n[{title}]")
    print("-" * 50)


def list_safetensor_files(model_id: str) -> tuple[list[str], list[str], list[str]]:
    """
    List all safetensor files in the repository.

    Returns:
        Tuple of (old_3shard_files, new_4shard_files, other_files)
    """
    all_files = list_repo_files(model_id)
    safetensor_files = sorted(f for f in all_files if f.endswith('.safetensors'))

    old_files = [f for f in safetensor_files if "-of-00003.safetensors" in f]
    new_files = [f for f in safetensor_files if "-of-00004.safetensors" in f]
    other_files = [f for f in safetensor_files if f not in old_files + new_files]

    return old_files, new_files, other_files


def get_referenced_files(model_id: str) -> set[str]:
    """Get files referenced by model.safetensors.index.json (the authoritative source)."""
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return set(index.get("weight_map", {}).values())


def extract_tensor_info(model_id: str, filenames: list[str]) -> dict[str, TensorInfo]:
    """
    Download checkpoint files and extract tensor keys with their shapes.

    Returns:
        Dict mapping tensor key -> TensorInfo(filename, shape)
    """
    tensors = {}
    for filename in filenames:
        filepath = hf_hub_download(model_id, filename)
        with safe_open(filepath, framework="pt") as sf:
            for key in sf.keys():
                shape = tuple(sf.get_tensor(key).shape)
                tensors[key] = TensorInfo(filename=filename, shape=shape)
    return tensors


def find_shape_conflicts(
    old_tensors: dict[str, TensorInfo],
    new_tensors: dict[str, TensorInfo],
) -> list[ShapeConflict]:
    """Find keys that exist in both old and new files with different shapes."""
    conflicts = []
    overlapping_keys = set(old_tensors.keys()) & set(new_tensors.keys())

    for key in sorted(overlapping_keys):
        old_info = old_tensors[key]
        new_info = new_tensors[key]
        if old_info.shape != new_info.shape:
            conflicts.append(ShapeConflict(key=key, old=old_info, new=new_info))

    return conflicts


def main():
    print_header(f"HuggingFace Checkpoint Redundancy Validation\nModel: {MODEL_ID}")

    # Step 1: List files in repository
    print_section("1. Files in Repository")
    old_files, new_files, other_files = list_safetensor_files(MODEL_ID)

    print(f"Old 3-shard files: {len(old_files)}")
    for f in old_files:
        print(f"  - {f}")

    print(f"\nNew 4-shard files: {len(new_files)}")
    for f in new_files:
        print(f"  - {f}")

    print(f"\nOther files: {other_files}")

    # Step 2: Check which files are referenced by the index
    print_section("2. Authoritative Index (model.safetensors.index.json)")
    referenced_files = get_referenced_files(MODEL_ID)
    print(f"Referenced files: {sorted(referenced_files)}")

    orphaned_files = set(old_files + new_files) - referenced_files
    print(f"\nOrphaned files (NOT in index):")
    for f in sorted(orphaned_files):
        print(f"  - {f}  ← SHOULD BE DELETED")

    # Step 3: Analyze tensor shapes
    print_section("3. Tensor Shape Analysis")

    print("Downloading and analyzing old 3-shard files...")
    old_tensors = extract_tensor_info(MODEL_ID, old_files)

    print("Downloading and analyzing new 4-shard files...")
    new_tensors = extract_tensor_info(MODEL_ID, new_files)

    overlapping_count = len(set(old_tensors.keys()) & set(new_tensors.keys()))
    print(f"\nKeys in old files: {len(old_tensors)}")
    print(f"Keys in new files: {len(new_tensors)}")
    print(f"Overlapping keys:  {overlapping_count}")

    # Step 4: Find shape conflicts
    print_section("4. Shape Conflicts (Root Cause of Loading Errors)")
    conflicts = find_shape_conflicts(old_tensors, new_tensors)

    if conflicts:
        print(f"Found {len(conflicts)} conflicting keys:\n")
        for conflict in conflicts:
            print(f"  {conflict.key}:")
            print(f"    OLD: {conflict.old.shape}  ({conflict.old.filename})")
            print(f"    NEW: {conflict.new.shape}  ({conflict.new.filename})")
            print()
    else:
        print("No shape conflicts found.")

    # Conclusion
    print_header("CONCLUSION")

    if orphaned_files:
        print("[CONFIRMED] Repository contains orphaned checkpoint files.")
        print(f"  - {len(orphaned_files)} files not referenced by index.json")
        print(f"  - {overlapping_count} duplicate keys between old and new files")

    if conflicts:
        print(f"\n[CRITICAL] {len(conflicts)} shape conflicts detected.")
        print("  - FMS loads ALL .safetensors files via glob")
        print("  - Old files have output_dim=42, new files have output_dim=256")
        print("  - This causes: RuntimeError: size mismatch (256 vs 42)")

    if orphaned_files or conflicts:
        print("\n[RECOMMENDATION FOR IBM]")
        print("  Delete these orphaned files from the HuggingFace repository:")
        for f in sorted(orphaned_files):
            print(f"    huggingface-cli repo delete-file {MODEL_ID} {f}")

    print()
    return bool(orphaned_files), bool(conflicts)


if __name__ == "__main__":
    main()
