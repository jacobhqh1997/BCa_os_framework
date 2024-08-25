from torch import nn
from torch.nn import Module


class LatentTransformer(Module):
    def __init__(self, get_latent_attn, get_latent_ff, num_latent_blocks_per_layer,
                 weight_tie_layers):
        super().__init__()
        self.latent_blocks = nn.ModuleList([])
        self.num_latent_blocks_per_layer = num_latent_blocks_per_layer
        for latent_block_index in range(num_latent_blocks_per_layer):
            should_cache = latent_block_index > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}
            self.latent_blocks.append(nn.ModuleList([
                get_latent_attn(**cache_args, name=f"latent_attn_{latent_block_index}"),
                get_latent_ff(**cache_args, name=f"latent_ff_{latent_block_index}")]))

    def forward(self, x):
        for latent_attn, latent_ff in self.latent_blocks:
            x = latent_attn(x) + x
            x = latent_ff(x) + x
        return x


def build_perceiver_layers(layers, depth, get_cross_attn, get_cross_ff,
                           get_latent_attn, get_latent_ff,
                           weight_tie_layers,
                           num_latent_blocks_per_layer=1,
                           ):
    for i in range(depth):
        should_cache = i > 0 and weight_tie_layers
        cache_args = {'_cache': should_cache}
        layers.append(nn.ModuleList([
            get_cross_attn(**cache_args, name="cross_attn"),
            get_cross_ff(**cache_args, name="cross_ff"),
            LatentTransformer(get_latent_attn, get_latent_ff,
                              num_latent_blocks_per_layer=num_latent_blocks_per_layer,
                              weight_tie_layers=weight_tie_layers)]))
