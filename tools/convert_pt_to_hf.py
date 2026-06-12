import argparse

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert RLinf full_weights.pt for Qwen2.5-VL to HuggingFace format."
    )
    parser.add_argument(
        "--base-model",
        default="/workspace/models/Qwen2.5-VL-3B-Instruct",
        help="Path to the original Qwen2.5-VL HuggingFace model directory.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to RLinf actor/model_state_dict/full_weights.pt.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Output HuggingFace model directory.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Maximum shard size passed to save_pretrained.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow missing or unexpected keys when loading the checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=not args.non_strict,
    )

    print(f"missing keys: {len(missing_keys)}")
    if missing_keys:
        print(missing_keys[:20])
    print(f"unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(unexpected_keys[:20])

    model.save_pretrained(
        args.save_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.save_dir)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.save_dir)
    print(f"saved HuggingFace model to {args.save_dir}")


if __name__ == "__main__":
    main()
