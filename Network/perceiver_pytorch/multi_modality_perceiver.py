from typing import Iterable, Dict, List

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn
from torch.nn import Identity

from perceiver_pytorch.caching import cache_by_name_fn
from perceiver_pytorch.modalities import InputModality, modality_encoding
from perceiver_pytorch.perceiver_pytorch import PreNorm, Attention, FeedForward, cache_fn, fourier_encode, \
    FeedForwardGELU
from perceiver_pytorch.common import build_perceiver_layers


# An implementation of Perceiver that can accept multiple data modalities in the same forward.
class MultiModalityPerceiver(nn.Module):
    def __init__(
            self,
            *,
            modalities: Iterable[InputModality],
            depth,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=None,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            num_latent_blocks_per_layer=1,
            use_gelu: bool = False,
    ):
        """

        :param modalities:
        :param depth: Number of times the perceiver will perform cross-attention between latent and input.
        :param num_latents:
        :param latent_dim:
        :param cross_heads:
        :param latent_heads:
        :param cross_dim_head:
        :param latent_dim_head:
        :param num_classes: Number of classes to predict, or if None, return the hidden state (num latents x hidden_dim)
        :param attn_dropout:
        :param ff_dropout:
        :param weight_tie_layers: True: share weights across layers, False no shared weights.
        :param num_latent_blocks_per_layer: Number of blocks in the latent transformer.
        :param use_gelu: Use GELU activation like the Perceiver preprint indicates. False,
               with Lucidrains' GEGLU activation in feed forward instead.

        """
        super().__init__()
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        self.max_modality_dim = input_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        ff_type = FeedForwardGELU if use_gelu else FeedForward
        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, ff_type(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, ff_type(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_by_name_fn, (
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        build_perceiver_layers(self.layers, depth, get_cross_attn, get_cross_ff,
                               get_latent_attn, get_latent_ff,
                               weight_tie_layers,
                               num_latent_blocks_per_layer=num_latent_blocks_per_layer)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, multi_modality_data: Dict[str, Tensor], mask=None):
        """

        :param data: a dictionary where keys are modality names and Tensor contain a batch
        of modality input data.
        :param mask:
        :return:
        """
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []
        linearized_data_per_layer: Dict[int, List[Tensor]] = {}

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert modality_name in self.modalities, f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _, device = *data.shape, data.device
            assert len(
                axis) == modality.input_axis, f'input data must have the right number of  for modality {modality_name}. ' \
                                              f'Expected {modality.input_axis} while forward argument offered {len(axis)}'
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos,
                                     modality.max_freq, modality.num_freq_bands, modality.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities

            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,)).to(device)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(b, axis, modality_index, num_modalities, device=device)

            to_concat = (data, padding, enc_pos, modality_encodings)

            data = torch.cat(to_concat, dim=-1)
            data = rearrange(data, 'b ... d -> b (...) d')
            linearized_data.append(data)
        b = batch_sizes.pop()
        x = repeat(self.latents, 'n d -> b n d', b=b)
        # for i, data in enumerate(linearized_data):
        #     print(f"Size of tensor {i}: {data.size()}")
        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)

        for cross_attn, cross_ff, latent_transformer in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            x = latent_transformer(x) + x
        x = self.pool(x)

        return self.to_logits(x)

    def pool(self, x):
        """
        Perform pooling over latents.
        :param x: batch x num_latents x latent_dim
        :return: pooled x
        """
        # implement global pooling
        return x.mean(dim=-2)


class MultiModalityPerceiverNoPooling(MultiModalityPerceiver):
    def __init__(self, *, modalities: Iterable[InputModality], depth,
                 num_latents=512, latent_dim=512, cross_heads=1,
                 latent_heads=8, cross_dim_head=64, latent_dim_head=64,
                 attn_dropout=0., ff_dropout=0.,
                 weight_tie_layers=False, num_latent_blocks_per_layer=1,
                 use_gelu: bool = True):
        """
        Perceiver that returns hidden state. Makes it possible to configure pooling with
        the result of forward.
        :param modalities:
        :param depth: Number of times the perceiver will perform cross-attention between latent and input.
        :param num_latents:
        :param latent_dim:
        :param cross_heads:
        :param latent_heads:
        :param cross_dim_head:
        :param latent_dim_head:
        :param attn_dropout:
        :param ff_dropout:
        :param weight_tie_layers: True: share weights across layers, False no shared weights.
        :param num_latent_blocks_per_layer: Number of blocks in the latent transformer.
        :param use_gelu: Use GELU activation like the Perceiver preprint indicates. False,
               with Lucidrains' GEGLU activation in feed forward instead.

        """

        super().__init__(modalities=modalities, depth=depth, num_latents=num_latents, latent_dim=latent_dim,
                         cross_heads=cross_heads, latent_heads=latent_heads, cross_dim_head=cross_dim_head,
                         latent_dim_head=latent_dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout,
                         weight_tie_layers=weight_tie_layers, num_latent_blocks_per_layer=num_latent_blocks_per_layer,
                         use_gelu=use_gelu, num_classes=1)
        self.to_logits = Identity()

    def pool(self, x):
        """
        Do not pool.
        :param x: batch x num_latents x latent_dim
        :return: pooled x
        """
        # no pooling
        return x
