"""
Merge LoRA Adapter → Full Model
Usage: python merge.py --adapter ../checkpoints/lora-adapter --output ../merged-model
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge(adapter_path: str, output_path: str, base_model: str = None):
    # Try to detect base model from adapter config
    if not base_model:
        import json
        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Could not detect base model. Pass --base explicitly.")

    print(f"Base model  : {base_model}")
    print(f"Adapter     : {adapter_path}")
    print(f"Output      : {output_path}\n")

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, trust_remote_code=True)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_path} ...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("\n✅ Merge complete!")
    print(f"   Merged model: {output_path}")
    print(f"   Next step: bash ../scripts/convert_gguf.sh {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,  help="Path to LoRA adapter dir")
    parser.add_argument("--output",  required=True,  help="Output path for merged model")
    parser.add_argument("--base",    default=None,   help="Base model name (auto-detected if not set)")
    args = parser.parse_args()

    merge(args.adapter, args.output, args.base)
