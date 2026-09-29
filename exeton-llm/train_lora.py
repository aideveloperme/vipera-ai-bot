"""
Step 5 (optional) — LoRA fine-tune an open LLM on the Exeton dataset, on DGX Spark.

DGX Spark has 128 GB of unified memory, so an 8B model trains comfortably in
bf16 with LoRA (no 4-bit quantisation needed). Larger bases (e.g. 70B) need
QLoRA and much longer runs.

Usage (inside the NGC PyTorch container, see README):
  python train_lora.py --base meta-llama/Llama-3.1-8B-Instruct
  python train_lora.py --base Qwen/Qwen2.5-7B-Instruct --epochs 2

Output:
  outputs/exeton-lora/          LoRA adapter
  outputs/exeton-merged/        full merged model (serve with vLLM / Ollama)
"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from knowledge import DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default=str(DATA_DIR / "sft.jsonl"))
    ap.add_argument("--out", default="outputs/exeton-lora")
    ap.add_argument("--merged-out", default="outputs/exeton-merged")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")

    # Prompt/completion format → loss only on the assistant answer, not the long CONTEXT.
    ds = load_dataset("json", data_files=args.data, split="train").map(
        lambda ex: {"prompt": ex["messages"][:-1], "completion": ex["messages"][-1:]},
        remove_columns=["messages"],
    ).train_test_split(test_size=0.05, seed=42)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=LoraConfig(
            r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            bf16=True,
            max_length=args.max_len,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"LoRA adapter saved → {args.out}")

    # Merge adapter into the base weights for simple serving with vLLM / Ollama.
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.merged_out, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_out)
    print(f"Merged model saved → {args.merged_out}")


if __name__ == "__main__":
    main()
