# Faithfulness-Aware Summarization of Scientific Papers using Large Language Models

Fine-tuning **FLAN-T5-small** with **LoRA (Low-Rank Adaptation)** to generate faithful, concise summaries of scientific abstracts — achieving a **+33% ROUGE-1 improvement** while updating only **2.25% of model parameters**.

---

## Overview

Scientific summarization demands that generated text remain factually grounded in the source document. This project explores parameter-efficient fine-tuning as a path toward both summarization quality and faithfulness, comparing a fully fine-tuned baseline against a LoRA-adapted model across ROUGE metrics and qualitative output analysis.

| System | Trainable Params | Trainable % | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| Base FLAN-T5-small | 76,961,152 | 100.00% | 1.4907 | 0.388 | 0.236 | 0.324 |
| LoRA FLAN-T5-small | 1,769,472 | **2.25%** | **0.402** | **0.718** | **0.602** | **0.669** |

> LoRA achieves significantly higher summarization quality at a fraction of the trainable parameter cost.

---

## Repository Structure

```
├── LoRA_Integrated_Faithfulness_Aware_Scientific_Summarization_using_Large_Language_Models.ipynb
├── LoRA-Integrated Faithfulness-Aware Scientific Summarization using Large Language Models.pdf
├── Presentation_Demo_Video.mp4
├── README.md
└── Results/
    ├── results_base_model/
    │   ├── best_model/          # Full fine-tuned FLAN-T5-small weights
    │   └── loss_log.txt
    └── results_final/
        ├── lora_adapter/        # LoRA adapter weights (adapter_model.safetensors)
        ├── base_vs_lora_comparison.csv
        ├── base_vs_lora_rouge.png
        ├── lora_example_outputs.csv
        ├── loss_plot.png
        └── sample_outputs_base_vs_lora.csv
```

> Model weights (`.safetensors`) and the demo video are tracked via **Git LFS**.

---

## Pipeline

```
Raw JSONL dataset
       │
       ▼
Tokenization (SentencePiece, max 512 tokens)
       │
       ├──► Base Fine-Tuning (FLAN-T5-small, 3 epochs, lr=2e-5)
       │           │
       │           ▼
       │    ROUGE Evaluation + Summary Generation
       │
       └──► LoRA Adaptation (r=16, α=32, 15 epochs, lr=3e-4)
                   │
                   ▼
            ROUGE Evaluation + Error Analysis
```

---

## Model Configuration

### Base Model
- **Architecture:** `google/flan-t5-small` (encoder-decoder, seq2seq)
- **Tokenizer:** SentencePiece — vocabulary size ~32,100

### LoRA Configuration
| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`α`) | 32 |
| Dropout | 0.05 |
| Target modules | `q`, `k`, `v`, `o`, `wi`, `wo` |
| Task type | `SEQ_2_SEQ_LM` |

### Training Hyperparameters
| Parameter | Base Model | LoRA Model |
|---|---|---|
| Epochs | 3 | 15 |
| Learning rate | 2e-5 | 3e-4 |
| Batch size | 4 | 4 |
| Optimizer | AdamW | AdamW |
| Max input length | 512 | 512 |
| Max output length | 96 | 96 |

### Decoding Strategy
| Parameter | Value |
|---|---|
| Beam search (`num_beams`) | 3 |
| Min length | 25 |
| Repetition penalty | 1.8 |
| No-repeat n-gram size | 3 |
| Length penalty | 1.2 |

---

## Quick Start

### Requirements

```bash
pip install transformers datasets sentencepiece evaluate rouge_score accelerate peft
```

### Run on Google Colab

1. Open `LoRA_Integrated_Faithfulness_Aware_Scientific_Summarization_using_Large_Language_Models.ipynb` in Google Colab.
2. Run all cells sequentially.
3. Outputs are saved to `Results/results_final/`.

### Load the LoRA Adapter (Inference)

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
tokenizer  = AutoTokenizer.from_pretrained("google/flan-t5-small")
model      = PeftModel.from_pretrained(base_model, "Results/results_final/lora_adapter")

inputs = tokenizer("Your scientific abstract here...", return_tensors="pt", max_length=512, truncation=True)
output = model.generate(**inputs, num_beams=3, min_length=25, max_new_tokens=96)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## Sample Outputs

| Input (excerpt) | Reference Summary | LoRA Output |
|---|---|---|
| *"...structured prompting to improve reliability and reduce errors in generated scientific summaries..."* | *"...improves phrase-level overlap with reference summaries, although long inputs require truncation."* | *"...improves phrase-level overlap with reference summaries, and shows that long inputs require truncation."* |
| *"...human-in-the-loop error analysis can support domain adaptation..."* | *"...demonstrates improved summarization behavior despite the limitation that performance remains limited by dataset size."* | *"...shows that it improves alignment between generated and reference summaries, while noting that performance remains limited by dataset size."* |

---

## Reproducibility

All experiments use fixed random seeds across Python, NumPy, and PyTorch. Dataset splits, hyperparameters, model checkpoints, and evaluation outputs are saved under `Results/`.

Configuration snapshot: `Results/results_final/run_config.json`

---

## Limitations

- Dataset is synthetic; real-world scientific abstracts may exhibit different distributions.
- Small dataset size limits generalization.
- ROUGE metrics measure lexical overlap but do not directly capture factual consistency or hallucination.

---

## Author

**Mushfiqur Rahman Rafin**  
Graduate Student, Computer Science  
University of Missouri, Kansas City
