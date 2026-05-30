"""
QLoRA Fine-Tuning Script
Usage: python train.py [--config config.json]

Trains a LoRA adapter on top of a base model using 4-bit quantization.
Your data should be in ../data/train.jsonl with fields: instruction, output
(optionally: input, system)
"""

import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Model
    base_model: str          = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str          = "../checkpoints/lora-adapter"

    # Data
    train_file: str          = "../data/train.jsonl"
    val_file: Optional[str]  = None           # auto-split if None
    val_split: float         = 0.05

    # LoRA
    lora_r: int              = 16             # rank
    lora_alpha: int          = 32
    lora_dropout: float      = 0.05
    target_modules: list     = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    epochs: int              = 3
    batch_size: int          = 4
    grad_accumulation: int   = 4             # effective batch = batch_size × grad_acc
    learning_rate: float     = 2e-4
    max_seq_length: int      = 1024
    warmup_ratio: float      = 0.05
    lr_scheduler: str        = "cosine"
    fp16: bool               = True
    save_steps: int          = 100
    eval_steps: int          = 100
    logging_steps: int       = 10


def load_config(path: Optional[str]) -> TrainConfig:
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return TrainConfig(**data)
    return TrainConfig()


# ── Prompt Template ───────────────────────────────────────────────────────────
def format_prompt(sample: dict, tokenizer) -> str:
    """Format a JSONL sample into an instruction-tuning prompt."""
    system = sample.get("system", "You are a helpful assistant.")
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    output = sample.get("output", "")

    if inp:
        user_content = f"{instruction}\n\n{inp}"
    else:
        user_content = instruction

    # ChatML format (works across most models)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\n{'='*60}")
    print("QLoRA Fine-Tuning")
    print(f"  Base model : {cfg.base_model}")
    print(f"  Train data : {cfg.train_file}")
    print(f"  Output dir : {cfg.output_dir}")
    print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit Quantization ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Base Model ────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── LoRA Config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={"train": cfg.train_file})

    if cfg.val_file:
        dataset["validation"] = load_dataset("json", data_files=cfg.val_file)["train"]
    else:
        split = dataset["train"].train_test_split(test_size=cfg.val_split, seed=42)
        dataset["train"]      = split["train"]
        dataset["validation"] = split["test"]

    print(f"Train samples : {len(dataset['train'])}")
    print(f"Val samples   : {len(dataset['validation'])}\n")

    def formatting_fn(batch):
        return [format_prompt(s, tokenizer) for s in
                [{k: batch[k][i] for k in batch} for i in range(len(batch["instruction"]))]]

    # ── Training Args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accumulation,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        report_to="none",
        group_by_length=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_fn,
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=4,
    )

    print("Starting training...\n")
    trainer.train()

    print(f"\nSaving adapter to {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("\n✅ Training complete!")
    print(f"   Adapter saved to: {cfg.output_dir}")
    print(f"   Next step: python merge.py --adapter {cfg.output_dir}")


if __name__ == "__main__":
    main()
