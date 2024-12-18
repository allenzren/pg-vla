import glob
import logging
import os
from typing import Optional, Tuple

import hydra
import torch
from safetensors import safe_open
from torch import nn

from src.model.paligemma.utils import JointKVCache, KVCache
from src.model.vla.modules import ActionTimeEncoder
from src.model.vla.utils import sample_from_transformed_beta
from src.utils.monitor import log_execution_time

log = logging.getLogger(__name__)


class VLA(nn.Module):
    @log_execution_time()
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.pad_token_id = cfg.pad_token_id
        self.image_token_index = cfg.image_token_index
        self.num_proprio_tokens = cfg.cond_steps
        self.num_action_tokens = cfg.horizon_steps
        self.image_text_hidden_size = cfg.image_text_hidden_size
        self.proprio_hidden_size = cfg.proprio_hidden_size
        self.action_hidden_size = cfg.action_hidden_size
        self.cache_block_indices = [
            0,
            1,
        ]  # do not cache action since not autoregressive, which is the last block of the three
        self.use_lm_head = cfg.get("use_lm_head", False)

        # Action parameterization
        self.num_inference_steps = cfg.num_inference_steps
        self.horizon_steps = cfg.horizon_steps
        self.action_dim = cfg.action_dim
        self.final_action_clip_value = cfg.final_action_clip_value
        self.sig_min = cfg.get("flow_sig_min", 0.001)
        self.gamma_alpha = cfg.get("flow_gamma_alpha", 1.5)
        self.gamma_beta = cfg.get("flow_gamma_beta", 1)
        self.gamma_max = 1 - self.sig_min
        self.schedule = cfg.get("flow_schedule", "gamma")
        assert self.schedule in [
            "linear",
            "gamma",
        ], f"Invalid schedule: {self.schedule}"

        # text input only
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size,
            self.image_text_hidden_size,
            self.pad_token_id,
        )  # 0.527B parameters

        # Vision
        self.vision_tower = hydra.utils.instantiate(cfg.vision)
        self.multi_modal_projector = hydra.utils.instantiate(cfg.vision_projector)

        # lm + action expert
        self.joint_model = hydra.utils.instantiate(cfg.joint)

        # Action, proprio, time encoders
        self.action_time_encoder = ActionTimeEncoder(
            cfg.action_dim,
            cfg.action_hidden_size,
        )
        self.proprio_encoder = nn.Linear(
            cfg.proprio_dim,
            cfg.proprio_hidden_size,
        )

        # Action decoder
        self.action_decoder = nn.Linear(
            cfg.action_hidden_size,
            cfg.action_dim,  # full chunk
        )

        # optional text output
        if self.use_lm_head:
            self.lm_head = nn.Linear(
                self.image_text_hidden_size,
                self.vocab_size,
                bias=False,
            )

    @property
    def action_expert_parameters(self):
        action_expert_parameters = []
        for name, param in self.joint_model.named_parameters():
            if (
                "q_projs.2" in name
                or "k_projs.2" in name
                or "v_projs.2" in name
                or "o_projs.2" in name
                or "mlp.2" in name
                or "layernorms.2" in name
                or "layernorm.2" in name
                or "action_norm" in name
            ):
                action_expert_parameters.append(param)
        return (
            list(self.action_time_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + action_expert_parameters
        )

    @log_execution_time()
    def load_pretrained_weights(self):
        """vision, projector, lm from paligemma"""

        # load tensors from files
        safetensors_files = glob.glob(
            os.path.join(self.cfg.pretrained_model_path, "*.safetensors")
        )
        tensors = {}
        for safetensors_file in safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        # load embed tokens
        embed_tokens_state_dict = self.embed_tokens.state_dict()
        for k, v in tensors.items():
            if "embed_tokens" in k:
                new_key = k.replace("language_model.model.embed_tokens.", "")
                embed_tokens_state_dict[new_key] = v
        self.embed_tokens.load_state_dict(embed_tokens_state_dict, strict=True)
        print("Loaded pre-trained weights for embed tokens")

        # load vision tower --- "vision_tower.vision_model" -> "vision_model"
        vision_tower_state_dict = self.vision_tower.state_dict()
        for k, v in tensors.items():
            if "vision_tower" in k:
                new_key = k.replace("vision_tower.", "")
                vision_tower_state_dict[new_key] = v
        self.vision_tower.load_state_dict(vision_tower_state_dict, strict=True)
        print("Loaded pre-trained weights for vision tower")

        # load projector --- "multi_modal_projector.linear" -> "linear"
        multi_modal_projector_state_dict = self.multi_modal_projector.state_dict()
        for k, v in tensors.items():
            if "multi_modal_projector" in k:
                new_key = k.replace("multi_modal_projector.", "")
                multi_modal_projector_state_dict[new_key] = v
        self.multi_modal_projector.load_state_dict(
            multi_modal_projector_state_dict, strict=True
        )
        print("Loaded pre-trained weights for projector")

        # load lm --- account for blocks
        joint_model_state_dict = self.joint_model.state_dict()
        for k, v in tensors.items():
            if "language_model.model" in k:
                new_key = k.replace("language_model.model.", "")
                # "self_attn.o_proj.weight" -> "self_attn.o_projs.0.weight"
                if (
                    "q_proj" in new_key
                    or "v_proj" in new_key
                    or "k_proj" in new_key
                    or "o_proj" in new_key
                ):
                    new_key = new_key.replace("proj.", "projs.0.")
                # "mlp.up_proj.weight" -> "mlp.0.up_proj.weight"
                elif "mlp" in new_key:
                    new_key = new_key.replace("mlp.", "mlp.0.")
                # "input_layernorm.weight" -> "input_layernorms.0.weight"
                elif "layernorm" in new_key:
                    new_key = new_key.replace("layernorm.", "layernorms.0.")
                # "norm.weight" -> "norm.weight"  # no change needed
                elif "norm" in new_key:
                    pass
                # skip "embed_tokens"
                elif "embed_tokens" in new_key:
                    continue
                joint_model_state_dict[new_key] = v
        self.joint_model.load_state_dict(joint_model_state_dict, strict=False)
        print("Loaded pre-trained weights for lm part of the joint model")

    def freeze_embedding(self):
        self.embed_tokens.weight.requires_grad = False

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def _merge_input_ids_with_pixel_values(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Extract input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.embed_tokens(input_ids)

        # TODO: cache this in action inference
        # Extract image features from siglip and projector
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Normalize the image features
        _, _, embed_dim = image_features.shape
        bsz, seq_len = input_ids.shape
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.image_text_hidden_size**0.5)

        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(
            bsz,
            seq_len,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )  # already padded since initialized as pad_token_id
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.image_token_index) & (
            input_ids != self.pad_token_id
        )
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.image_token_index

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        return final_embedding

    def _build_causal_mask_and_position_ids_for_action(
        self,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.LongTensor]]:
        """blocks"""
        device = attention_mask.device
        num_image_text_tokens = torch.sum(attention_mask, dim=1)
        bsz, max_num_image_text_tokens = attention_mask.shape
        total_len = (
            max_num_image_text_tokens + self.num_proprio_tokens + self.num_action_tokens
        )
        proprio_start = max_num_image_text_tokens
        proprio_end = max_num_image_text_tokens + self.num_proprio_tokens
        action_start = proprio_end

        # block attention
        causal_mask = torch.full(
            (bsz, total_len, total_len),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=device,
        )  # smallest value, avoid using inf for softmax nan issues with padding
        for idx, num in enumerate(num_image_text_tokens):
            causal_mask[idx, :num, :num] = 0  # image/text attend to itself
            causal_mask[idx, proprio_start:proprio_end, :num] = (
                0  # proprio attend to image/text
            )
            causal_mask[idx, action_start:, :num] = 0  # action attend to image/text
        causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
            0  # proprio attend to itself
        )
        causal_mask[:, action_start:, proprio_start:] = (
            0  # action attend to itself and proprio
        )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        # position ids for each blocks, ignore padding
        position_ids_all = (
            torch.arange(max_num_image_text_tokens)[None].expand(bsz, -1).to(device),
            torch.arange(self.num_proprio_tokens)[None].expand(bsz, -1).to(device),
            torch.arange(self.num_action_tokens)[None].expand(bsz, -1).to(device),
        )

        return causal_mask, position_ids_all

    def _build_causal_mask_and_position_ids_for_text(
        self,
        q_len: int,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        dtype, device = attention_mask.dtype, attention_mask.device

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1, "Using KV cache so should only use one single token"
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (bsz, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return causal_mask, position_ids

    @torch.no_grad()
    def infer_action(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        proprios: torch.FloatTensor,
        kv_cache: Optional[JointKVCache] = None,
    ) -> torch.FloatTensor:
        kv_cache = JointKVCache() if kv_cache is None else kv_cache

        # Merge the text tokens and the image tokens
        inputs_embeds = self._merge_input_ids_with_pixel_values(input_ids, pixel_values)

        # Build causal mask and position ids for action
        (
            causal_mask,
            position_ids_all,
        ) = self._build_causal_mask_and_position_ids_for_action(attention_mask)

        # Encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # Sample pure action noise
        bsz = input_ids.size(0)
        action = torch.randn(
            (
                bsz,
                self.horizon_steps,
                self.action_dim,
            ),
            device=input_ids.device,
        )

        # forward euler integration
        # no need to update causal_mask or position_ids_all
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(bsz, device=input_ids.device)
        for _ in range(self.num_inference_steps):
            # encode action and time into embedding
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.action_time_encoder(action, t)
            # forward thru joint model with block attention
            # [Batch_Size, Horizon_Steps, Embed_Dim]
            action_embeds = self.joint_model(
                attention_mask=causal_mask,
                position_ids_all=position_ids_all,
                inputs_embeds=inputs_embeds,
                proprio_embeds=proprio_embeds,
                action_embeds=action_embeds,
                kv_cache=kv_cache,
                cache_block_indices=self.cache_block_indices,
            )
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action_vel = self.action_decoder(action_embeds)
            action += delta_t * action_vel
            t += delta_t

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action

    @torch.no_grad()
    def infer_text(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # Merge the text tokens and the image tokens
        inputs_embeds = self._merge_input_ids_with_pixel_values(input_ids, pixel_values)

        # Build causal mask and position ids for text
        q_len = input_ids.size(1)
        (
            attention_mask,
            position_ids,
        ) = self._build_causal_mask_and_position_ids_for_text(
            q_len, attention_mask, kv_cache
        )

        hidden_states = self.joint_model.forward_text_only(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(hidden_states).float()
        return_data = {
            "logits": logits,
        }
        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        return return_data

    # ---------- Flow matching training ----------#

    def psi_t(
        self,
        x: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.sig_min) * t) * x + t * x1

    def forward(
        self,
        pixel_values: torch.ByteTensor,
        input_ids: torch.LongTensor,
        proprios: torch.FloatTensor,
        actions: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        """diffusion / flow matching loss for action prediction, no cache"""
        device = input_ids.device
        bsz = len(input_ids)
        x1 = actions
        if self.schedule == "linear":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (
                torch.rand(1, device=device) + torch.arange(bsz, device=device) / bsz
            ) % (1 - eps)
        elif self.schedule == "gamma":  # from pi0 paper
            t = sample_from_transformed_beta(
                self.gamma_alpha, self.gamma_beta, self.gamma_max, size=bsz
            )
            t = torch.from_numpy(t).float().to(device)  # (B,)
        # x ~ p_t(x0)
        x0 = torch.randn_like(actions, device=device)

        # get noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        psi_t = self.psi_t(x0, x1, t)

        # Merge the text tokens and the image tokens
        inputs_embeds = self._merge_input_ids_with_pixel_values(input_ids, pixel_values)

        # Build causal mask and position ids for action
        (
            causal_mask,
            position_ids_all,
        ) = self._build_causal_mask_and_position_ids_for_action(attention_mask)

        # Encode proprio
        proprio_embeds = self.proprio_encoder(proprios)

        # inference with noisy action
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        action_embeds = self.action_time_encoder(psi_t, t)
        # forward thru joint model with block attention
        # [Batch_Size, Horizon_Steps, Embed_Dim]
        action_embeds = self.joint_model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            inputs_embeds=inputs_embeds,
            proprio_embeds=proprio_embeds,
            action_embeds=action_embeds,
            cache_block_indices=self.cache_block_indices,
        )
        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # Compare to true velocity
        d_psi = x1 - (1 - self.sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


if __name__ == "__main__":
    import argparse

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    from src.model.vla.processing import VLAProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--load_pretrained_weights", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--loss_only", action="store_true")
    args = parser.parse_args()
    assert not (args.text_only and args.loss_only)

    config = OmegaConf.load("config/train/pg_oxe.yaml")
    if args.text_only:
        config.use_lm_head = True
    device = "cpu" if args.cpu else "cuda"
    model = VLA(config).to(device)
    if args.load_pretrained_weights:
        model.load_pretrained_weights()
        if args.text_only:
            model.tie_weights()

    # dummy image --- replace the first image with a real one
    bsz = 1 if args.text_only else 2
    dummy_images = torch.randint(0, 256, (bsz, 3, 224, 224))  # not used if text_only
    real_image_path = "media/maniskill_pp.png"
    real_image = Image.open(real_image_path).convert("RGB")
    real_image_t = torch.from_numpy(
        np.array(real_image.resize((224, 224))).transpose(2, 0, 1)
    )
    dummy_images[0] = real_image_t

    # text and proprio
    dummy_texts = [
        "this image shows ",
        "this is a nice portrait of London because ",
    ][:bsz]
    dummy_proprio = torch.rand(bsz, config.cond_steps, config.action_dim).to(
        device
    )  # not used if text_only

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_path, padding_side="right"
    )
    assert tokenizer.padding_side == "right"

    # processor
    num_image_tokens = config.vision.config.num_image_tokens
    processor = VLAProcessor(tokenizer, num_image_tokens, config.max_seq_len)

    # process image and text
    model_inputs = processor(text=dummy_texts, images=dummy_images)
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(
        device
    )  # with padding if bsz > 1
    pixel_values = model_inputs["pixel_values"].to(device)

    # inference - text or actions
    if args.text_only:
        # no sampling
        kv_cache = KVCache()
        num_tokens_to_generate = 10
        print(f"Generating text of {num_tokens_to_generate} tokens...")

        # Generate tokens until you see the stop token
        stop_token = processor.tokenizer.eos_token_id
        generated_tokens = []

        for _ in range(num_tokens_to_generate):
            outputs = model.infer_text(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)  # Remove batch dimension
            generated_tokens.append(next_token)
            # Stop if the stop token has been generated
            if next_token.item() == stop_token:
                break
            # Append the next token to the input --- use cache so only the new token
            input_ids = next_token.unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
            )
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        # Decode the generated tokens
        decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("Prompt:", dummy_texts[0])
        print("Generated text:", decoded)
    elif args.loss_only:
        dummy_actions = torch.randn(bsz, config.horizon_steps, config.action_dim).to(
            device
        )
        loss = model.loss(
            pixel_values=pixel_values,
            input_ids=input_ids,
            proprios=dummy_proprio,
            actions=dummy_actions,
            attention_mask=attention_mask,
        )
        print("Loss:", loss)
    else:  # dummy action generation
        actions = model.infer_action(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,  # only for image/text
            proprios=dummy_proprio,
        )
        print("Final action dimensions:", actions.shape)
        print("Final action values:", actions)
