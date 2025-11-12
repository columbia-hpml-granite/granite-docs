---
slug: /
title: Granite Speech Integration in FMS - Columbia University HPML
sidebar_position: 1
hide_title: true
---

# Granite Speech Integration in the Foundation Model Stack (FMS)

## Columbia University High Performance Machine Learning

### In Collaboration with IBM Research

This is the technical documentation site for the Granite Speech integration project, conducted as part of Columbia University's High Performance Machine Learning course in collaboration with IBM Research.

## 📚 Project Overview

**Project Title:** Granite Speech Integration in the Foundation Model Stack (FMS)

**Project Objective:** Integrate the Granite Speech model into IBM's Foundation Model Stack so it can run end-to-end under `torch.compile`, evaluate its performance against eager execution, and document what makes a speech model compile-efficient inside FMS.

### Project Details

- **Duration:** 8 Weeks
- **Institution:** Columbia University
- **Partner:** IBM Research
- **Focus:** High Performance Speech Model Integration & Optimization

## 👥 Team

**Faculty Supervisors:**

- Dr. Kaoutar El Maghraoui
- Dr. Rashed Bhatti

**Student Team Members:**

- [Aneesh Durai](https://github.com/aneeshdurai)
- [Geonsik Moon](https://github.com/gsmoon97)
- [In Keun Kim](https://github.com/nearKim)
- [Zachary Zusin](https://github.com/zacharyzusin)

## 🎯 About Granite Speech

Granite Speech is IBM's open-source speech-aware large language model designed for automatic speech recognition (ASR) and automatic speech translation (AST).

### Key Features

- **Model Size:** 8 billion parameters (Granite Speech 3.3 8B)
- **Languages:** English, French, German, Spanish, Portuguese
- **Translation:** X-En and En-X pairs including Japanese and Mandarin
- **Architecture:** Two-stage design combining speech encoder with language model

### Architecture Components

1. **Speech Encoder:** 16 conformer blocks with CTC training
2. **Speech Projector:** 2-layer window query transformer with 5x downsampling
3. **Language Model:** Granite 3.3 8B instruct with 128k context length
4. **LoRA Adapters:** Rank-64 fine-tuning on query/value projections

## 📅 Weekly Updates

Track our progress throughout the 8-week project:

- **[Week 1](weekly/week1)** - Project Orientation & Implementation Planning
- **[Week 2](weekly/week2)** - Profiling & Performance Analysis
- **[Week 3](weekly/week3)** - TBD

## 🔗 Resources

- **Granite Speech Model:** [HuggingFace](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- **IBM Granite Docs:** [Documentation](https://www.ibm.com/granite/docs/models/speech)
- **Research Paper:** [arXiv:2505.08699](https://arxiv.org/pdf/2505.08699)
- **Project Repository:** [GitHub](https://github.com/your-org/granite-docs)

## 🎓 Technical Background

### Foundation Model Stack (FMS)

IBM's Foundation Model Stack is a framework designed to efficiently integrate and deploy foundation models with optimizations for production environments, including support for `torch.compile` for improved performance.

### Project Goals

1. Successfully integrate Granite Speech model into FMS
2. Enable end-to-end execution under `torch.compile`
3. Benchmark performance: compile vs. eager execution
4. Identify and document compile-efficiency patterns for speech models
5. Optimize encoder and audio-to-text fusion pathways
