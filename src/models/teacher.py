"""Qwen3-VL-8B-Instruct teacher: image-only embedding extraction."""

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen3-VL needs a recent transformers with Qwen3VLForConditionalGeneration "
        "(e.g. pip install -U 'transformers>=4.57.0' — Qwen3-VL needs Qwen3VLForConditionalGeneration). "
        "Do not load Qwen3 checkpoints into Qwen2_5_VL — weights will not match."
    ) from exc


def load_teacher(model_name: str = "Qwen/Qwen3-VL-8B-Instruct", dtype=torch.bfloat16):
    """Load the teacher model and processor.  Returns (model, processor)."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


@torch.no_grad()
def extract_embeddings(
    model,
    processor,
    image_paths: list[str],
    prompt: str = "Describe this place.",
    max_pixels: int = 448 * 448,
    min_pixels: int = 224 * 224,
) -> torch.Tensor:
    """Extract L2-normalised image embeddings from the teacher.

    Builds a single-image message per path, runs a forward pass, and
    mean-pools over the image token positions in the last hidden state.

    Returns:
        Tensor of shape (len(image_paths), hidden_dim).
    """
    all_embeds = []

    for path in image_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{path}",
                     "max_pixels": max_pixels, "min_pixels": min_pixels},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, D)

        # Identify image token positions via the processor's image token id
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mask = (inputs.input_ids == image_token_id).unsqueeze(-1)  # (1, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeds.append(F.normalize(pooled.squeeze(0).float(), p=2, dim=-1))

    return torch.stack(all_embeds)
