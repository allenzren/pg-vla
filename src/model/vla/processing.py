from typing import List, Optional

import torch

IMAGENET_STANDARD_MEAN = torch.tensor([0.5, 0.5, 0.5])
IMAGENET_STANDARD_STD = torch.tensor([0.5, 0.5, 0.5])


def add_image_tokens_to_prompt(
    prefix_prompt,
    bos_token,
    image_seq_len,
    image_token,
):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: torch.LongTensor,
    scale: float,
) -> torch.FloatTensor:
    rescaled_image = image * scale
    return rescaled_image


def normalize(
    image: torch.LongTensor,
    mean: torch.FloatTensor,
    std: torch.FloatTensor,
) -> torch.FloatTensor:
    assert image.ndim == 4, f"Expected 4D tensor, got {image.ndim}D tensor."
    assert (
        image.shape[1] == 3
    ), f"Expected 3 channels at axis 1, got {image.shape[1]} channels."
    mean = mean[None, :, None, None]  # add batch and spatial dimensions
    std = std[None, :, None, None]
    image = (image - mean) / std
    return image


def process_images(
    images: torch.LongTensor,
    rescale_factor: float,
    image_mean: torch.FloatTensor,
    image_std: torch.FloatTensor,
) -> torch.FloatTensor:
    # Rescale the pixel values to be in the range [0, 1]
    images = rescale(images, scale=rescale_factor)

    # Normalize the images to have mean 0 and standard deviation 1
    images = normalize(images, mean=image_mean, std=image_std)

    return images


class VLAProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        tokenizer,
        num_image_tokens: int,
        max_seq_len: Optional[int] = None,
        tokenizer_padding: str = "longest",
    ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.max_seq_len = max_seq_len
        self.tokenizer_padding = tokenizer_padding

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # We will add the BOS and EOS tokens ourselves
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: torch.LongTensor,
        truncation: bool = True,
    ) -> dict:
        assert len(images) == len(
            text
        ), f"Received {len(images)} images for {len(text)} prompts."
        assert (
            images.dtype == torch.uint8
        ), f"Expected uint8 tensor for images, got {images.dtype}."

        pixel_values = process_images(
            images,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # Prepend a `self.image_seq_length` number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding=self.tokenizer_padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
