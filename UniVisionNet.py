from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
import torch

macro_modality = InputModality(
    name='macro',
    input_channels=1,  # number of channels for each token of the input
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)
micro_modality = InputModality(
    name='micro',
    input_channels=24,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
perceiver_multi = MultiModalityPerceiver(
    modalities=(macro_modality, micro_modality),
    depth=8,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=1,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True,
    num_latent_blocks_per_layer=6  # Note that this parameter is 1 in the original Lucidrain implementation
    # whether to weight tie layers (optional, as indicated in the diagram)
)

