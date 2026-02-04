#!/usr/bin/env python3
"""
Heretic Abliteration for Qwen3-VL models.

This script applies abliteration to Qwen3-VL vision-language models by targeting
only the language model component while freezing/ignoring the vision encoder.

Model structure:
    Qwen3VLForConditionalGeneration
        └── model (Qwen3VLModel)
              ├── visual (Qwen3VLVisionModel) ← IGNORED
              └── language_model (Qwen3VLTextModel)
                    └── layers ← TARGET
        └── lm_head
"""

import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


def load_prompts(dataset_name, split, column):
    """Load prompts from a HuggingFace dataset."""
    dataset = load_dataset(dataset_name, split=split)
    return [row[column] for row in dataset]


def get_layers(model):
    """
    Get transformer layers from any supported model architecture.

    Supports:
    - Qwen3VLForConditionalGeneration: model.model.language_model.layers
    - Mistral3ForConditionalGeneration: model.language_model.layers
    - AutoModelForCausalLM: model.model.layers
    """
    # Qwen3-VL structure
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return model.model.language_model.layers
    # Mistral3 structure
    elif hasattr(model, 'language_model'):
        return model.language_model.layers
    # Standard CausalLM structure
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    else:
        raise ValueError(f"Unknown model structure: {type(model)}")


def get_residuals(model, tokenizer, prompts, batch_size=8):
    """Get residual vectors for prompts at the first generated token position."""
    all_residuals = []
    layers = get_layers(model)
    device = layers[0].self_attn.o_proj.weight.device

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]

        # Format as chat
        chats = [[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}
        ] for p in batch]

        chat_prompts = [tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in chats]

        inputs = tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Hidden states for the first generated token
        hidden_states = outputs.hidden_states[0]

        # Stack hidden states from all layers
        residuals = torch.stack(
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        all_residuals.append(residuals.to(torch.float32).cpu())
        print(f"  Processed {min(i+batch_size, len(prompts))}/{len(prompts)} prompts")

    return torch.cat(all_residuals, dim=0)


def abliterate(model, refusal_directions, direction_index: Optional[float], parameters: dict):
    """
    Apply abliteration to the language model weights.

    This modifies the attention output projections and MLP down projections
    to remove the refusal direction from the model's representation space.
    """
    layers = get_layers(model)
    n_modified = 0

    if direction_index is None:
        refusal_direction = None
    else:
        import math
        weight, index = math.modf(direction_index + 1)
        refusal_direction = F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1],
                weight,
            ),
            p=2,
            dim=0,
        )

    for layer_index in range(len(layers)):
        layer = layers[layer_index]

        for component, params in parameters.items():
            if component == "attn.o_proj":
                matrices = [layer.self_attn.o_proj.weight]
            elif component == "mlp.down_proj":
                matrices = [layer.mlp.down_proj.weight]
            else:
                continue

            distance = abs(layer_index - params.max_weight_position)

            if distance > params.min_weight_distance:
                continue

            weight = params.max_weight + (distance / params.min_weight_distance) * (
                params.min_weight - params.max_weight
            )

            if refusal_direction is None:
                layer_refusal_direction = refusal_directions[layer_index + 1]
            else:
                layer_refusal_direction = refusal_direction

            for matrix in matrices:
                mat_device = matrix.device
                mat_dtype = matrix.dtype

                projector = torch.outer(
                    layer_refusal_direction.to(mat_device),
                    layer_refusal_direction.to(mat_device),
                ).to(dtype=mat_dtype)

                matrix.sub_(weight * (projector @ matrix))
                n_modified += 1

    print(f"  Abliteration complete: modified {n_modified} matrices")


def main():
    parser = argparse.ArgumentParser(description="Heretic Abliteration for Qwen3-VL")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct", help="Model path or HF repo")
    parser.add_argument("--output", default="/home/aibox/heretic/output/Qwen3-VL-32B-heretic", help="Output path")
    parser.add_argument("--good-prompts", type=int, default=400, help="Number of harmless prompts")
    parser.add_argument("--bad-prompts", type=int, default=400, help="Number of harmful prompts")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for residual calculation")
    parser.add_argument("--direction-index", type=float, default=None,
                        help="Direction index (None for per-layer, or float like 25.0)")
    args = parser.parse_args()

    print("=" * 70)
    print("HERETIC ABLITERATION - Qwen3-VL")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Load the vision-language model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Get layer information
    layers = get_layers(model)
    n_layers = len(layers)
    print(f"Model loaded. Language model layers: {n_layers}")
    print(f"Layer 0 device: {layers[0].self_attn.o_proj.weight.device}")

    # Verify the vision encoder exists but we're not touching it
    if hasattr(model, 'model') and hasattr(model.model, 'visual'):
        print(f"Vision encoder present (frozen/ignored)")

    # Load prompts
    print()
    print("Loading prompts...")
    good_prompts = load_prompts("mlabonne/harmless_alpaca", f"train[:{args.good_prompts}]", "text")
    bad_prompts = load_prompts("mlabonne/harmful_behaviors", f"train[:{args.bad_prompts}]", "text")
    print(f"  Good prompts: {len(good_prompts)}")
    print(f"  Bad prompts: {len(bad_prompts)}")

    # Calculate refusal directions
    print()
    print("Calculating refusal directions...")
    print("  Getting residuals for good prompts...")
    good_residuals = get_residuals(model, tokenizer, good_prompts, batch_size=args.batch_size)
    print("  Getting residuals for bad prompts...")
    bad_residuals = get_residuals(model, tokenizer, bad_prompts, batch_size=args.batch_size)

    device = layers[0].self_attn.o_proj.weight.device
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
        p=2,
        dim=1,
    ).to(device)

    print(f"  Refusal directions shape: {refusal_directions.shape}")

    # Apply abliteration
    print()
    print("=" * 70)
    print("APPLYING ABLITERATION")
    print("=" * 70)

    # Scale parameters based on layer count (original was tuned for 32-layer models)
    # Qwen3-VL-32B has 64 layers, so we scale positions accordingly
    scale = n_layers / 32.0

    # Parameters scaled from successful trials
    parameters = {
        "attn.o_proj": AbliterationParameters(
            max_weight=0.94,
            max_weight_position=24.58 * scale,
            min_weight=0.51,
            min_weight_distance=12.72 * scale,
        ),
        "mlp.down_proj": AbliterationParameters(
            max_weight=1.08,
            max_weight_position=20.82 * scale,
            min_weight=0.82,
            min_weight_distance=10.75 * scale,
        ),
    }

    # Default direction index scaled to model size
    direction_index = args.direction_index
    if direction_index is None:
        # Use a direction from mid-upper layers (similar to trial 154)
        direction_index = 15.94 * scale

    print(f"  Direction index: {direction_index}")
    print(f"  Layer scale factor: {scale}")
    print()

    abliterate(model, refusal_directions, direction_index, parameters)

    # Save model
    print()
    print(f"Saving model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done!")

    print()
    print("=" * 70)
    print("Model saved successfully!")
    print(f"Output: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
