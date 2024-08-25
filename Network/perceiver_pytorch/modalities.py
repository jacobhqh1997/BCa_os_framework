from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Embedding


@dataclass
class InputModality:
    name: str
    input_channels: int
    input_axis: int
    num_freq_bands: int
    max_freq: float
    freq_base: int = 2

    @property
    def input_dim(self) -> int:
        # Calculate the dimension of this modality.
        input_dim = self.input_axis * ((self.num_freq_bands * 2) + 1) + self.input_channels
        return input_dim


def modality_encoding(batch_size: int, axes, modality_index: int, num_modalities: int,
                      device=torch.device('cpu')) -> Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.
    :param modality_index:
    :param num_modalities:
    :return:
    """
    one_hot = torch.eye(num_modalities, num_modalities, device=device)[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    to_expand.append(num_modalities)

    one_hot = one_hot.expand(to_expand)
    return one_hot


@dataclass
class InputModalityWithEmbedding(InputModality):
    embedding: Embedding = None

    def embedding_dim(self, depth: int) -> int:
        if not self.embedding:
            return self.input_dim
        else:
            # each layer sees a subset of the embedding output.
            pos_encoding_dim = ((self.num_freq_bands * 2) + 1)
            return self.embedding.embedding_dim // depth + pos_encoding_dim

    def embedding_for_layer(self, embedded: Tensor, layer_index: int, depth: int):
        if not self.embedding:
            # This modality does not require embedding, we return the features:
            return embedded
        assert self.input_axis == 1, "embedding for layer is not supported with axis !=1"
        assert embedded.dim()==3, "embedded text tensor must have 3 dimensions: B x L x D"
        # embedded has dimension B x L x D, B: batch, L: sequence length, D: full embedding dim.
        dim_per_layer = embedded.size(-1) // depth
        start_dim_index = layer_index * dim_per_layer
        end_dim_index = (layer_index+1) * dim_per_layer
        return embedded[:, :, start_dim_index:end_dim_index]

    def maybe_embed(self, data: Tensor):
        if self.embedding:
            return self.embedding(data.squeeze(2))
        else:
            return data
