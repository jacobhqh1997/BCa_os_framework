from typing import Iterable, Dict, List

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn
from torch.nn import ModuleDict

from perceiver_pytorch.caching import cache_by_name_fn
from perceiver_pytorch.common import build_perceiver_layers
from perceiver_pytorch.modalities import InputModalityWithEmbedding, modality_encoding
from perceiver_pytorch.perceiver_pytorch import PreNorm, Attention, FeedForward, fourier_encode


# An implementation of Perceiver that can accept multiple data modalities in the same forward, including
# modalities which requires embedding.
class MultiModalityWithTextPerceiver(nn.Module):
    def __init__(
            self,
            *,
            modalities: Iterable[InputModalityWithEmbedding],
            depth,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1000,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
            num_latent_blocks_per_layer=6
    ):
        super().__init__()
        self.depth = depth
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # Register any embeddings inside this torch module:
        self.embeddings = ModuleDict({modality.name: modality.embedding for modality
                                      in modalities if hasattr(modality, 'embedding') and
                                      modality.embedding})

        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.embedding_dim(self.depth) for modality in modalities) + modality_encoding_dim
        self.max_modality_dim = input_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_by_name_fn, (
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        build_perceiver_layers(self.layers, depth, get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff,
                               weight_tie_layers, num_latent_blocks_per_layer=num_latent_blocks_per_layer)

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

        linearized_data_per_layer: Dict[int, List[Tensor]] = {}

        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            assert modality_name in self.modalities, f"modality {modality_name} was not defined in constructor"
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _, device = *data.shape, data.device
            assert len(axis) == \
                   modality.input_axis, f'input data must have the right number of axes for modality {modality_name}. ' \
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
            padding_size = self.max_modality_dim - modality.embedding_dim(self.depth) - num_modalities
            current_data_modality_shape_without_channels = data.size()[0:-1]
            padding = torch.zeros(size=current_data_modality_shape_without_channels + (padding_size,),
                                  device=data.device)
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(b, axis, modality_index, num_modalities, device=device)

            if modality_name in self.embeddings:
                # restore modality embedding from this torch module:
                modality.embedding = self.embeddings[modality_name]
            data = modality.maybe_embed(data)

            for i in range(self.depth):
                layer_data = modality.embedding_for_layer(data, i, self.depth)
                to_concat = (layer_data, padding, enc_pos, modality_encodings)

                layer_data = torch.cat(to_concat, dim=-1)
                layer_data = rearrange(layer_data, 'b ... d -> b (...) d')

                if i not in linearized_data_per_layer:
                    linearized_data_per_layer[i] = []
                linearized_data_per_layer[i].append(layer_data)

        b = batch_sizes.pop()
        x = repeat(self.latents, 'n d -> b n d', b=b)

        for i, (cross_attn, cross_ff, latent_transformer) in enumerate(self.layers):
            # Concatenate all the modalities:
            data = torch.cat(linearized_data_per_layer[i], dim=1)
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            x = latent_transformer(x) + x

        x = x.mean(dim=-2)
        return self.to_logits(x)
