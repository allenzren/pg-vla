import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.lora import get_layer
from src.model.paligemma.modules import (
    AdaptiveLayerscale,
    AdaptiveRMSNorm,
    GemmaMLP,
    GemmaRMSNorm,
    GemmaRotaryEmbedding,
)
from src.model.paligemma.utils import (
    JointKVCache,
    KVCache,
    apply_rotary_pos_emb,
    repeat_kv,
)

# from torch.nn.attention.flex_attention import flex_attention
# flex_attention = torch.compile(flex_attention, mode="max-autotune")
# # introduced in gemma
# def softclamp_score_mod(value):
#     def identity(score, b, h, q, k):
#         return score

#     def softclamped(score, b, h, q, k):
#         score = score / value
#         score = torch.tanh(score)
#         score = score * value
#         return score

#     return softclamped if value > 0.0 else identity


class JointModel(nn.Module):
    def __init__(self, config, quantize: bool = False, lora: bool = False):
        super().__init__()
        config.quantize = quantize
        config.lora = lora
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        (
            self.image_text_hidden_size,
            self.proprio_hidden_size,
            self.action_hidden_size,
        ) = config.hidden_sizes

        # transformer layers
        self.layers = nn.ModuleList(
            [
                JointDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # final norms
        if config.get("use_lm_head", False):
            self.norm = GemmaRMSNorm(
                self.image_text_hidden_size, eps=config.rms_norm_eps
            )
        if config.action_expert_adaptive_mode:
            self.action_norm = AdaptiveRMSNorm(
                self.action_hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )  # time has the same dim as action
        else:
            self.action_norm = GemmaRMSNorm(
                self.action_hidden_size, eps=config.rms_norm_eps
            )

    def _check_action_parameter_by_name(self, name: str) -> bool:
        if (
            "q_projs.2" in name
            or "k_projs.2" in name
            or "v_projs.2" in name
            or "o_projs.2" in name
            or "mlp.2" in name
            or "layernorms.2" in name
            or "action_norm" in name
            or "_scale" in name  # adaptive layerscale
        ):
            return True
        return False

    def _check_gemma_trainable_parameter_by_name(self, name: str) -> bool:
        last_hidden_layer_index = self.num_hidden_layers - 1
        if not (
            "q_projs.2" in name
            or "k_projs.2" in name
            or "v_projs.2" in name
            or "o_projs.2" in name
            or "mlp.2" in name
            or "layernorms.2" in name
            or "action_norm" in name
            or "_scale" in name  # adaptive layerscale
            or name == "norm.weight"  # no need to tune final norm
            or f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_projs" in name
            or f"{last_hidden_layer_index}.self_attn.v_projs"
            in name  # no need to tune part of last layer
        ):
            return True
        return False

    def _check_gemma_unused_parameter_by_name(self, name: str) -> bool:
        last_hidden_layer_index = self.num_hidden_layers - 1
        if not self._check_action_parameter_by_name(name) and (
            f"{last_hidden_layer_index}.post" in name
            or f"{last_hidden_layer_index}.mlp" in name
            or f"{last_hidden_layer_index}.self_attn.o_projs" in name
            or f"{last_hidden_layer_index}.self_attn.v_projs" in name
        ):
            return True
        return False

    @property
    def action_parameters(self):
        action_parameters = []
        for name, param in self.named_parameters():
            if self._check_action_parameter_by_name(name):
                action_parameters.append(param)
        return action_parameters

    @property
    def trainable_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.named_parameters():
            if self._check_gemma_trainable_parameter_by_name(name):
                gemma_parameters.append(param)
        return gemma_parameters

    @property
    def trainable_lora_gemma_parameters(self):
        gemma_parameters = []
        for name, param in self.named_parameters():
            if self._check_gemma_trainable_parameter_by_name(name):
                if "lora_" in name:
                    gemma_parameters.append(param)
        return gemma_parameters

    def freeze_non_lora_weights_in_gemma(self):
        for name, param in self.named_parameters():
            if self._check_gemma_trainable_parameter_by_name(name):
                param.requires_grad = True if "lora_" in name else False

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: Tuple[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        proprio_embeds: torch.FloatTensor,
        action_embeds: torch.FloatTensor,
        time_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[JointKVCache] = None,
        cache_block_indices: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(
            self.image_text_hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        # normalize proprio and action too
        proprio_normalizer = torch.tensor(
            self.proprio_hidden_size**0.5, dtype=proprio_embeds.dtype
        )
        proprio_hidden_states = proprio_embeds * proprio_normalizer
        action_normalizer = torch.tensor(
            self.action_hidden_size**0.5, dtype=action_embeds.dtype
        )
        action_hidden_states = action_embeds * action_normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states, proprio_hidden_states, action_hidden_states = decoder_layer(
                (
                    hidden_states,
                    proprio_hidden_states,
                    action_hidden_states,
                ),
                attention_mask=attention_mask,
                position_ids_all=position_ids_all,
                time_embeds=time_embeds,
                kv_cache=kv_cache,
                cache_block_indices=cache_block_indices,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        # if isinstance(self.action_norm, AdaptiveRMSNorm):
        #     action_hidden_states = self.action_norm(action_hidden_states, time_embeds)
        # else:
        #     action_hidden_states = self.action_norm(action_hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def forward_text_only(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(
            self.image_text_hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer.forward_text_only(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        return hidden_states


class JointDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.final_layer = layer_idx == config.num_hidden_layers - 1
        self.self_attn = JointAttention(config=config, layer_idx=layer_idx)

        self.mlp = nn.ModuleList()
        for block_index, (hidden_size, intermediate_size) in enumerate(
            zip(config.hidden_sizes, config.intermediate_sizes, strict=False)
        ):
            mlp_config = config
            config.hidden_size = hidden_size
            config.intermediate_size = intermediate_size
            if block_index != 0:  # only quantize or lora for image/text block
                quantize, lora = False, False
            else:
                quantize, lora = config.quantize, config.lora
            self.mlp.append(GemmaMLP(mlp_config, quantize=quantize, lora=lora))

        self.input_layernorms = nn.ModuleList()
        self.post_attention_layernorms = nn.ModuleList()
        for block_index, hidden_size in enumerate(config.hidden_sizes):
            if config.action_expert_adaptive_mode and block_index == 2:
                self.input_layernorms.append(
                    AdaptiveRMSNorm(
                        hidden_size, config.time_hidden_size, eps=config.rms_norm_eps
                    )
                )
                self.post_attention_layernorms.append(
                    AdaptiveRMSNorm(
                        hidden_size, config.time_hidden_size, eps=config.rms_norm_eps
                    )
                )
                if config.action_expert_adaptive_mode == "adaLN-Zero":
                    self.post_adaptive_scale = AdaptiveLayerscale(
                        hidden_size, config.time_hidden_size
                    )
                    self.final_adaptive_scale = AdaptiveLayerscale(
                        hidden_size, config.time_hidden_size
                    )
            else:
                self.input_layernorms.append(
                    GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
                )
                self.post_attention_layernorms.append(
                    GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
                )
        self.action_expert_adaptive_mode = config.action_expert_adaptive_mode

    def forward(
        self,
        hidden_states_all: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids_all: Tuple[torch.Tensor],
        time_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[JointKVCache] = None,
        cache_block_indices: Optional[Tuple[int]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        residuals = hidden_states_all
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_pre = []
        for hidden_states, layernorm in zip(
            hidden_states_all, self.input_layernorms, strict=False
        ):
            if isinstance(layernorm, AdaptiveRMSNorm):
                hidden_states_pre.append(layernorm(hidden_states, time_embeds))
            else:
                hidden_states_pre.append(layernorm(hidden_states))

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_all = self.self_attn(
            hidden_states_all=tuple(hidden_states_pre),
            attention_mask=attention_mask,
            position_ids_all=position_ids_all,
            kv_cache=kv_cache,
            cache_block_indices=cache_block_indices,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_pre_res = []
        for block_index, (residual, hidden_states) in enumerate(
            zip(residuals, hidden_states_all, strict=False)
        ):
            # if self.final_layer and block_index != 2:
            #     hidden_states_pre_res.append(None)
            if hasattr(self, "post_adaptive_scale") and block_index == 2:
                hidden_states_pre_res.append(
                    residual + self.post_adaptive_scale(hidden_states, time_embeds)
                )
            else:
                hidden_states_pre_res.append(residual + hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        residuals = hidden_states_pre_res
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_post = []
        for _, (hidden_states, layernorm) in enumerate(
            zip(hidden_states_pre_res, self.post_attention_layernorms, strict=False)
        ):
            # if self.final_layer and block_index != 2:
            #     hidden_states_post.append(None)
            if isinstance(layernorm, AdaptiveRMSNorm):
                hidden_states_post.append(layernorm(hidden_states, time_embeds))
            else:
                hidden_states_post.append(layernorm(hidden_states))

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_mlp = []
        for _, (hidden_states, mlp) in enumerate(
            zip(hidden_states_post, self.mlp, strict=False)
        ):
            # if self.final_layer and block_index != 2:
            #     hidden_states_mlp.append(None)
            # else:
            hidden_states_mlp.append(mlp(hidden_states))

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_final = []
        for block_index, (residual, hidden_states) in enumerate(
            zip(residuals, hidden_states_mlp, strict=False)
        ):
            # if self.final_layer and block_index != 2:
            #     hidden_states_final.append(None)
            if hasattr(self, "final_adaptive_scale") and block_index == 2:
                hidden_states_final.append(
                    residual + self.final_adaptive_scale(hidden_states, time_embeds)
                )
            else:
                hidden_states_final.append(residual + hidden_states)

        return tuple(hidden_states_final)

    def forward_text_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """Assume text is the first block"""
        residuals = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorms[0](hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.self_attn.forward_text_only(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states += residuals

        # [Batch_Size, Seq_Len, Hidden_Size]
        residuals = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorms[0](hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp[0](hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states += residuals

        return hidden_states


class JointAttention(nn.Module):
    """assume head_dim same for all blocks"""

    def __init__(
        self,
        config,
        layer_idx: Optional[int] = None,
        attn_softclamp: float = 50.0,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.final_layer = layer_idx == config.num_hidden_layers - 1
        self.hidden_sizes = config.hidden_sizes

        self.attn_softclamp = attn_softclamp
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        for hidden_size in self.hidden_sizes:
            assert hidden_size % self.num_heads == 0

        # only quantize or lora for image/text block
        image_text_layer = get_layer(
            config.quantize, config.lora, r=config.lora_r, dropout=config.lora_dropout
        )
        layers = [image_text_layer] + [nn.Linear for _ in range(2)]
        self.q_projs = nn.ModuleList(
            [
                layer(
                    hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for layer, hidden_size in zip(layers, self.hidden_sizes, strict=False)
            ]
        )
        self.k_projs = nn.ModuleList(
            [
                layer(
                    hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for layer, hidden_size in zip(layers, self.hidden_sizes, strict=False)
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                layer(
                    hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for layer, hidden_size in zip(layers, self.hidden_sizes, strict=False)
            ]
        )
        self.o_projs = nn.ModuleList(
            [
                layer(
                    self.num_heads * self.head_dim,
                    hidden_size,
                    bias=config.attention_bias,
                )
                for layer, hidden_size in zip(layers, self.hidden_sizes, strict=False)
            ]
        )
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states_all: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids_all: Tuple[torch.LongTensor],
        kv_cache: Optional[JointKVCache] = None,
        cache_block_indices: Optional[Tuple[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = len(hidden_states_all[0])
        q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all]
        num_blocks = len(hidden_states_all)

        # always re-compute queires
        query_states_all = []
        for block_idx in range(num_blocks):
            hidden_states = hidden_states_all[block_idx]
            q_len = q_lens[block_idx]
            # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
            query_states = self.q_projs[block_idx](hidden_states)
            # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
            query_states = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            query_states_all.append(query_states)

        # for keys and values, use the cache for some blocks
        key_states_all = []
        value_states_all = []
        flag_cached = kv_cache is not None and kv_cache.has_item(self.layer_idx)
        if flag_cached:
            key_states_cached, value_states_cached = kv_cache.get(self.layer_idx)
        for block_idx in range(num_blocks):
            if flag_cached and block_idx in cache_block_indices:
                # TODO: make this nicer
                cache_block_idx = cache_block_indices.index(block_idx)
                key_states_all.append(key_states_cached[cache_block_idx])
                value_states_all.append(value_states_cached[cache_block_idx])
            else:
                hidden_states = hidden_states_all[block_idx]
                q_len = q_lens[block_idx]
                # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
                key_states = self.k_projs[block_idx](hidden_states)
                # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
                key_states = key_states.view(
                    bsz, q_len, self.num_key_value_heads, self.head_dim
                ).transpose(1, 2)
                key_states_all.append(key_states)

                # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
                value_states = self.v_projs[block_idx](hidden_states)
                # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
                value_states = value_states.view(
                    bsz, q_len, self.num_key_value_heads, self.head_dim
                ).transpose(1, 2)
                value_states_all.append(value_states)

        # apply rotary embeddings
        for block_idx in range(num_blocks):
            query_states = query_states_all[block_idx]
            key_states = key_states_all[block_idx]
            value_states = value_states_all[block_idx]
            position_ids = position_ids_all[block_idx]
            # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
            cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
            # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            query_states_all[block_idx] = query_states
            key_states_all[block_idx] = key_states

        # only update when the cache is empty
        flag_to_cache = kv_cache is not None and not kv_cache.has_item(self.layer_idx)
        if flag_to_cache:
            kv_cache.update(
                [key_states_all[block_idx] for block_idx in cache_block_indices],
                [value_states_all[block_idx] for block_idx in cache_block_indices],
                self.layer_idx,
            )

        # Repeat the key and values to match the number of heads of the query
        for block_idx in range(num_blocks):
            key_states_all[block_idx] = repeat_kv(
                key_states_all[block_idx], self.num_key_value_groups
            )
            value_states_all[block_idx] = repeat_kv(
                value_states_all[block_idx], self.num_key_value_groups
            )

        # Concatenate all the blocks along sequence
        # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
        # [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim]
        query_states = torch.cat(query_states_all, dim=-2)
        key_states = torch.cat(key_states_all, dim=-2)
        value_states = torch.cat(value_states_all, dim=2)

        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Soft capping
        attn_weights = attn_weights / self.attn_softclamp
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.attn_softclamp

        # Apply the softmax
        attn_weights = attn_weights + attention_mask
        # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).type_as(query_states)
        # Apply the dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, sum(q_lens), self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, sum(q_lens), self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, sum(q_lens), -1)

        # split for blocks
        attn_outputs = torch.split(attn_output, q_lens, dim=1)

        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_outputs_final = []
        for block_idx in range(num_blocks):
            # if self.final_layer and block_idx != 2:
            #     attn_output = None
            # else:
            attn_output = self.o_projs[block_idx](attn_outputs[block_idx])
            attn_outputs_final.append(attn_output)

        return tuple(attn_outputs_final)

    def forward_text_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Assume text is the first block"""
        bsz, q_len, _ = hidden_states.size()  # [Batch_Size, Seq_Len, Hidden_Size]

        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_projs[0](hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_projs[0](hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_projs[0](hidden_states)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(
                key_states, value_states, self.layer_idx
            )

        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # # Flex attention --- https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459
        # score_mod_fn = softclamp_score_mod(self.attn_softclamp)
        # causal_attention = partial(
        #     flex_attention,
        #     # block_mask=attention_mask,  # not set up yet
        #     score_mod=score_mod_fn,
        # )
        # attn_output = causal_attention(query_states, key_states, value_states)

        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Soft capping
        attn_weights = attn_weights / self.attn_softclamp
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.attn_softclamp

        # Apply the softmax with masking
        attn_weights = attn_weights + attention_mask
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).type_as(query_states)
        # Apply the dropout
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        # Multiply by the values. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_projs[0](attn_output)

        return attn_output


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: torch.FloatTensor,
        max_period: float = 10000.0,
    ) -> torch.FloatTensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int, time_cond: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        if time_cond:
            self.linear_2 = nn.Linear(2 * width, width)
        else:
            self.linear_2 = nn.Linear(width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)
        self.time_cond = time_cond

    def forward(
        self,
        action: torch.FloatTensor,
        time_emb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        if self.time_cond:
            # repeat time embedding for seq_len
            # [Batch_Size, Seq_Len, Width]
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
            emb = torch.cat([time_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/train/pg_oxe.yaml")
    model = JointModel(cfg.joint.config)

    dummy_num_image_tokens = 7
    inputs_embeds = torch.randn(
        1,
        dummy_num_image_tokens,
        cfg.joint.config.hidden_sizes[0],
    )  # no history
    proprio_embeds = torch.randn(
        1,
        cfg.cond_steps,
        cfg.joint.config.hidden_sizes[1],
    )
    action_embeds = torch.randn(
        1,
        cfg.horizon_steps,
        cfg.joint.config.hidden_sizes[2],
    )
    q_lens = [
        dummy_num_image_tokens,
        cfg.cond_steps,
        cfg.horizon_steps,
    ]
    total_len = sum(q_lens)

    kv_cache = JointKVCache()
    cache_block_indices = [0, 1]
    position_ids_all = (
        torch.arange(dummy_num_image_tokens)[None],  # no text
        torch.arange(cfg.cond_steps)[None],
        torch.arange(cfg.horizon_steps)[None],
    )  # add batch dim

    # block attention
    causal_mask = torch.full(
        (total_len, total_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )  # smallest value
    causal_mask[:dummy_num_image_tokens, :dummy_num_image_tokens] = (
        0  # image/text attend to itself
    )
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + cfg.cond_steps
    causal_mask[proprio_start:proprio_end, :proprio_end] = (
        0  # proprio attend to itself and image/text
    )
    action_start = proprio_end
    causal_mask[action_start:, :] = 0  # action attend to itself and all

    # dummy denoising
    print("Initial action embeds", action_embeds)
    num_step = 3
    for _step in range(num_step):
        print("running dummy denoising step", _step)
        action_embeds = model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            inputs_embeds=inputs_embeds,
            proprio_embeds=proprio_embeds,
            action_embeds=action_embeds,
            kv_cache=kv_cache,
            cache_block_indices=cache_block_indices,
        )
        print("Updated action embeds", action_embeds)
