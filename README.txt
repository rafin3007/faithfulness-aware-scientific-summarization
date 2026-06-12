Project Title: Faithfulness-Aware Summarization of Scientific Papers using Large Language Models

Tokenizer
Model: FLAN-T5-small tokenizer (SentencePiece)
Vocabulary size: ~32,100

Tokenization:
Subword-based tokenization (splits rare and scientific terms)
Outputs: input_ids, attention_mask, labels
Implemented in dataset class using:
tokenizer(source_text, ...)
tokenizer(target_text, ...)

How to Run (Google Colab)
Install dependencies:
pip install transformers datasets sentencepiece evaluate rouge_score accelerate peft
Open notebook:
Faithfulness_Aware_Summarization_of_Scientific_Papers_using_Large_Language_Models.ipynb
Run all cells sequentially.

Key Hyperparameters

Base Model: FLAN-T5-small

LoRA Configuration:
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q", "k", "v", "o", "wi", "wo"]

Training:
Batch size: 4
Input length: 512
Output length: 96
Learning rate:
Base model: 2e-5
LoRA: 3e-4
Epochs:
Base: 3
LoRA: 15
Optimizer: AdamW

Decoding:
num_beams = 3
min_length = 25
repetition_penalty = 1.8
no_repeat_ngram_size = 3
length_penalty = 1.2

Pipeline Steps
Load JSONL dataset
Preprocess and tokenize text
Create DataLoader
Train model (forward + backward pass)
Evaluate (ROUGE)
Generate summaries
Evaluate base FLAN-T5 model
Apply LoRA adaptation
Train LoRA model
Evaluate using ROUGE metrics
Generate summaries
Save plots and results

Outputs Saved

Directory:
results_final/

Contains:
best_model/ → trained model
loss_log.txt → training loss
loss_plot.png → loss graph
lora_adapter/ → trained LoRA adapter
base_vs_lora_comparison.csv → comparison table
final_metrics.json → evaluation metrics
lora_example_outputs.csv → generated summaries
base_vs_lora_rouge.png → ROUGE comparison graph
error_analysis_examples.csv → qualitative analysis examples

Final Results

Base vs LoRA Comparison Table

System			Trainable %	Val Loss	ROUGE-1		ROUGE-2		ROUGE-L
Base FLAN-T5-small	100.0%		1.3800		0.3814		0.2336		0.3207
LoRA FLAN-T5-small	2.2475%		0.2381		0.7396		0.6172		0.6772

Key Observation:
LoRA significantly improves summarization quality while updating only ~2.25% of parameters.

Reproducibility
Fixed random seeds:
Python
NumPy
PyTorch
Saved:
dataset splits
hyperparameters
checkpoints
evaluation outputs

Configuration file:
results_final/run_config.json

Status
Pipeline works
Training stable
LoRA improves performance
Outputs meaningful and structured

Limitations
Synthetic dataset
Small dataset size
ROUGE does not fully measure factual consistency

Author
Mushfiqur Rahman Rafin
Graduate Student, Computer Science
University of Missouri, Kansas City