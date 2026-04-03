# SignalFlow: AI-Powered Anomaly Detection and Decision Intelligence

## Overview

SignalFlow is a modular AI system that transforms raw anomaly detection outputs into **structured, decision-ready signals** and **human-interpretable reasoning**.

The project combines:

- **PatchCore-based anomaly detection**
- a **registry layer** that stores structured model outputs as JSON
- an **LLM-based reasoning layer** that interprets those structured outputs

The main design idea is to separate **model computation** from **reasoning**:

- the perception model computes anomaly scores
- the registry layer structures those outputs
- the reasoning layer interprets the structured outputs

This makes the system easier to understand, easier to extend, and better aligned with future multimodal AI pipelines.

---

## Problem Statement

In real industrial environments, decisions are rarely made from a single output.

Systems often need to reason over:

- visual inspection outputs
- thresholds and calibration logic
- structured status signals
- future numeric or sensor-based signals
- operational context and policies

Most machine learning systems stop at prediction.

**SignalFlow explores what comes after prediction:**

- How can model outputs be converted into structured signals?
- How can those signals be passed into a reasoning layer in a controlled way?
- How can anomaly detection move toward decision intelligence?

---

## Core Design Principles

## 1. Models compute, they do not explain

The perception layer performs anomaly detection and produces technical outputs such as anomaly scores, predictions, and heatmaps.

## 2. LLMs reason, they do not see raw tensors

The reasoning layer operates on structured registry data, not raw feature maps, tensors, or embeddings.

## 3. Structured signals form the bridge

The registry JSON acts as the interface between perception outputs and downstream reasoning.

## 4. Decisions emerge through signal flow

The system is designed so that explanation is grounded in structured signals rather than unconstrained prompting.

---

## System Architecture

```text
Image -> PatchCore -> Anomaly Score -> Threshold / Calibration -> Registry JSON
      -> Structured Signals -> Reasoning Layer -> Explanation / Decision Support
```

## Project Vision

```text 
Prediction -> Structured Signals -> Explanation -> Decision Support
```

## Current Implementation Scope

The current implementation focuses on **image-based anomaly detection** and **structured reasoning** for the `juice_bottle` category from the **MVTec LOCO Anomaly Detection Dataset**.

### Implemented

- PatchCore-based anomaly detection
- image-level anomaly scoring
- threshold-based calibrated prediction logic
- registry JSON export
- heatmap and overlay generation
- raw prompt-based reasoning experiments
- LangChain structured reasoning experiments

### Planned / Conceptual Extensions

- multimodal signal aggregation
- sensor or tabular signal integration
- more robust deployment orchestration
- image-grounded or artifact-grounded reasoning
- multi-agent interpretation workflows


## Repository Structure

```text
SignalFlow-ai/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
├── src/
│ ├── patchcore_pipeline.py # Core anomaly detection pipeline
│ ├── export_registry.py # Output + metadata handling
│ ├── overlay_generator.py # Heatmap & overlay visualization
│ └── llm_reasoner.py # LLM-based reasoning layer
├── experiments/
│ ├── langchain_structured_explainer.py
│ ├── plain_prompt_baseline.py
│ ├── sarvam_iterative_test.py
│ └── juice_bottle_res256_.../ # Sample experiment output
├── samples/
│ ├── heatmaps/
│ └── overlays/
├── results/
└── .gitignore
```

## How the Project Works

### Step 1. Perception Layer

`src/patchcore_pipeline.py` runs PatchCore using Anomalib on the MVTec LOCO dataset.

This stage produces anomaly-related outputs such as:

* anomaly scores
* predicted labels
* anomaly maps / heatmaps
* ranked anomaly outputs

### Step 2. Registry Layer

`src/export_registry.py` exports model outputs into a structured registry JSON.

This registry acts as the core interface between perception and reasoning.

Typical registry fields include:

* image_id
* image_path
* pred_score
* pred_label
* pred_label_calibrated
* gt_label
* threshold_used
* rank

### Step 3. Visual Artifact Layer

`src/overlay_generator.py` generates overlays from images and heatmaps so results can be visually inspected.

A sample generated artifact folder is intentionally kept in the repository for demonstration.

### Step 4. Reasoning Layer

The project includes two reasoning styles:

- **raw prompt / API-based reasoning** via `src/llm_reasoner.py`
- **LangChain structured reasoning** via `experiments/langchain_structured_explainer.py`

This enables comparison between:

* loosely controlled prompt-based reasoning
* parser-backed structured reasoning over registry signals

## Main Files

These are the most important files to review first:

### Core pipeline
- src/patchcore_pipeline.py
- src/export_registry.py
- src/overlay_generator.py
- src/llm_reasoner.py

### Experiment scripts
- experiments/langchain_structured_explainer.py  (structured LLM reasoning)
- experiments/plain_prompt_baseline.py          (baseline comparison)
- experiments/sarvam_iterative_test.py          (iterative reasoning approach)

### Registry as the Bridge Layer

A central design idea in SignalFlow is that the registry is not just an output dump.

It is a signal interface between:

* the perception system
* threshold and ranking logic
* the reasoning layer

This makes the project closer to a system-design project than a pure model-training project.

Instead of giving raw model outputs directly to an LLM, SignalFlow first converts them into structured signals that are easier to interpret, validate, and extend.

### Raw Prompt vs Structured Reasoning

One of the key learnings from this project is the difference between prompt-only control and structured output control.

## Raw Prompt Reasoning

The prompt-only approach can:

* ignore strict JSON formatting rules
* confuse predicted labels with ground-truth labels
* produce verbose or inconsistent answers
* fail to follow rule-based anomaly logic consistently

## LangChain Structured Reasoning

The structured version improves reliability by using:

* PromptTemplate
* PydanticOutputParser
* schema-constrained outputs
* batch splitting for difficult registry input sizes

This makes the reasoning layer more controlled and closer to production-style output handling.

## Key Insight

A major insight from this project is:

formatting compliance and logical compliance must be evaluated separately.

A model may produce valid JSON while still failing on rule-based anomaly reasoning.

### Quick Start

1. Clone the repository

```bash
git clone <your-repo-url>
cd SignalFlow-ai
```

2. Install dependencies

Using pip:

```bash
pip install -r requirements.txt

```
Or using conda:

```bash
conda env create -f environment.yml
conda activate signalflow
```

3. Dataset

This project is built using the MVTec LOCO Anomaly Detection Dataset, specifically the juice_bottle category.

The dataset is not distributed with this repository.
Users must obtain it from the official dataset source and place it locally before running the project.

### Expected local path:

`data/mvtec_loco_anomaly_detection/juice_bottle/`


4. Run the anomaly detection pipeline
```bash
python src/patchcore_pipeline.py
```

5. Export the registry
```bash
python src/export_registry.py
```

6. Generate overlays
```bash
python src/overlay_generator.py
```

7. Run the reasoning layer

## Raw reasoning:

```bash
python src/llm_reasoner.py
```

## Structured LangChain comparison:

```bash
python experiments/langchain_structured_explainer.py
```
## Example Registry Output
```json
{
  "image_id": "004",
  "pred_score": 1.0,
  "pred_label_calibrated": 1,
  "gt_label": 0,
  "threshold_used": 0.995,
  "rank": 1
}
```

## Example Reasoning Output
```json
[
  {
    "image_id": "good/004",
    "predicted_status": "anomaly",
    "ground_truth_status": "normal",
    "is_correct": false,
    "error_type": "false_positive",
    "short_explanation": "Predicted anomaly, actual normal."
  }
]
```

### Why This Project Is Different

SignalFlow is not just a model demo.

It shows:

* anomaly detection
* structured signal engineering
* threshold-based decision logic
* registry design
* LLM reasoning over structured outputs
* comparison between unreliable prompt-only control and structured parsing


### Strengths
* combines computer vision and reasoning in one pipeline
* introduces a structured registry as a reusable system interface
* demonstrates system-design thinking beyond model training
* includes explainability-oriented artifacts such as overlays and structured outputs
* explores an important practical issue: LLM output reliability under reasoning constraints

### Limitations
* current implementation focuses on one image domain
* the system currently uses image-based signals only
* full multimodal signal aggregation is not yet implemented
* the reasoning layer depends on local or external LLM setup
* the project is still closer to a research prototype than a fully deployed production system

### Future Improvements
* expand the registry schema for multimodal signals
* integrate sensor and tabular signals
* add API or service-based deployment
* introduce evaluation for reasoning correctness at scale
* support image-grounded reasoning over heatmaps and overlays
* build orchestration for larger production workflows

### Tech Stack
* Python
* PyTorch
* Anomalib
* OpenCV
* JSON
* Ollama
* LangChain
* Pydantic


### Use Cases
* industrial quality inspection
* anomaly detection pipelines
* explainable AI workflows
* structured AI decision support
* multimodal monitoring system prototypes


### References

## Dataset

Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2022).
The MVTec LOCO AD Dataset: Towards Real-World Logical Anomaly Detection.
International Journal of Computer Vision, 130, 2684–2707.

## PatchCore

Roth, K., Pemula, L., Zepeda, J., Scholz, T., Dhillon, A., Nuske, S., & Hoshen, Y. (2022).
Towards Total Recall in Industrial Anomaly Detection.
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

## Anomalib

Anomalib: Open-Source Library for Deep Learning-Based Anomaly Detection.
Intel / Anomalib contributors.
