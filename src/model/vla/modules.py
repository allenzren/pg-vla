import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.paligemma.modules import GemmaMLP, GemmaRMSNorm, GemmaRotaryEmbedding
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
    def __init__(self, config):
        super().__init__()
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
        self.norm = GemmaRMSNorm(self.image_text_hidden_size, eps=config.rms_norm_eps)
        self.action_norm = GemmaRMSNorm(
            self.action_hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: Tuple[torch.LongTensor],
        inputs_embeds: torch.FloatTensor,
        proprio_embeds: torch.FloatTensor,
        action_embeds: torch.FloatTensor,
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
                kv_cache=kv_cache,
                cache_block_indices=cache_block_indices,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        action_hidden_states = self.action_norm(action_hidden_states)

        return action_hidden_states

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
        self.self_attn = JointAttention(config=config, layer_idx=layer_idx)

        self.mlp = nn.ModuleList()
        for hidden_size, intermediate_size in zip(
            config.hidden_sizes, config.intermediate_sizes, strict=False
        ):
            mlp_config = config
            config.hidden_size = hidden_size
            config.intermediate_size = intermediate_size
            self.mlp.append(GemmaMLP(mlp_config))

        self.input_layernorms = nn.ModuleList()
        self.post_attention_layernorms = nn.ModuleList()
        for hidden_size in config.hidden_sizes:
            self.input_layernorms.append(
                GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
            )
            self.post_attention_layernorms.append(
                GemmaRMSNorm(hidden_size, eps=config.rms_norm_eps)
            )

    def forward(
        self,
        hidden_states_all: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids_all: Tuple[torch.Tensor],
        kv_cache: Optional[JointKVCache] = None,
        cache_block_indices: Optional[Tuple[int]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        residuals = hidden_states_all
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_pre = []
        for hidden_states, layernorm in zip(
            hidden_states_all, self.input_layernorms, strict=False
        ):
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
        for residual, hidden_states in zip(residuals, hidden_states_all, strict=False):
            hidden_states_pre_res.append(residual + hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        residuals = hidden_states_pre_res
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_post = []
        for hidden_states, layernorm in zip(
            hidden_states_pre_res, self.post_attention_layernorms, strict=False
        ):
            hidden_states_post.append(layernorm(hidden_states))

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_mlp = []
        for hidden_states, mlp in zip(hidden_states_post, self.mlp, strict=False):
            hidden_states_mlp.append(mlp(hidden_states))

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_final = []
        for residual, hidden_states in zip(residuals, hidden_states_mlp, strict=False):
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

        self.q_projs = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    self.num_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for hidden_size in self.hidden_sizes
            ]
        )
        self.k_projs = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for hidden_size in self.hidden_sizes
            ]
        )
        self.v_projs = nn.ModuleList(
            [
                nn.Linear(
                    hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias,
                )
                for hidden_size in self.hidden_sizes
            ]
        )
        self.o_projs = nn.ModuleList(
            [
                nn.Linear(
                    self.num_heads * self.head_dim,
                    hidden_size,
                    bias=config.attention_bias,
                )
                for hidden_size in self.hidden_sizes
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
        ).to(query_states.dtype)
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
        ).to(query_states.dtype)
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
        x: torch.LongTensor,
    ) -> torch.FloatTensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionTimeEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(self, action_dim: int, width: int):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(width)
        self.linear_1 = nn.Linear(action_dim, width)
        self.linear_2 = nn.Linear(2 * width, width)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width)

    def forward(
        self,
        action: torch.FloatTensor,
        t: torch.LongTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Emb_Dim]
        time_emb = self.time_embedding(t)
        # repeat time embedding for seq_len
        # [Batch_Size, Seq_Len, Width]
        time_emb = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
        # [Batch_Size, Seq_Len, Width]
        action_emb = self.linear_1(action)
        time_action_emb = torch.cat([time_emb, action_emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(time_action_emb))
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
    causal_mask[
        :dummy_num_image_tokens, :dummy_num_image_tokens
    ] = 0  # image/text attend to itself
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + cfg.cond_steps
    causal_mask[
        proprio_start:proprio_end, :proprio_end
    ] = 0  # proprio attend to itself and image/text
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
