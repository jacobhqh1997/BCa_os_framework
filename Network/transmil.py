import numpy as np
import torch
import torch.nn as nn

from network.aggregator import BaseAggregator
from network.model_utils import PPEG, NystromTransformerLayer


class TransMIL(BaseAggregator):
    def __init__(self, num_classes, input_dim=1024, pos_enc='PPEG', **kwargs):
        super(BaseAggregator, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self.pos_enc = pos_enc
        print(f'Using {self.pos_enc} positional encoding')
        self._fc1 = nn.Sequential(
            nn.Linear(input_dim, 2048),nn.ReLU(),
            nn.Linear(2048, 1024),nn.ReLU(),
            nn.Linear(1024, 512),nn.ReLU())

        # self._fc1 = nn.Sequential(
        #     nn.Linear(input_dim, 1024),nn.ReLU(),
        #     nn.Linear(1024, 512),nn.ReLU())


        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.num_classes = num_classes
        self.layer1 = NystromTransformerLayer(dim=512)
        self.layer2 = NystromTransformerLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)
        

    def forward(self, x, coords=None):

        h = x 

        h = self._fc1(h)  

        #----> padding
        H = h.shape[1]
        if self.pos_enc == 'PPEG':
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H))) 
            add_length = _H * _W - H  
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  
        elif self.pos_enc == 'PPEG_padded': 
            if h.shape[1] > 1: 
                dimensions = coords.max(dim=1).values - coords.min(dim=1).values
                x_coords = coords[:, :, 1].unique(dim=1)  
                patch_size = (x_coords[:, 1:] - x_coords[:, :-1]).min(dim=-1).values
                offset = coords[:, 0, :] % patch_size
                dimensions_grid = ((dimensions - offset) / patch_size).squeeze(0) + 1
                _H, _W = dimensions_grid.int().tolist()
                base_grid = torch.zeros((h.shape[0], dimensions_grid[0].int().item(), dimensions_grid[1].int().item(), h.shape[-1]), device=h.device)
                grid_indices = (coords - offset.unsqueeze(1) - coords.min(dim=1).values.unsqueeze(1)) / patch_size
                grid_indices = grid_indices.long().cpu()
                base_grid[:, grid_indices[:, :, 0], grid_indices[:, :, 1]] = h.squeeze(0)
                h = base_grid.reshape((h.shape[0], -1, h.shape[-1]))
            else:
                _H, _W = 1, 1

        #----> cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #----> first translayer
        h = self.layer1(h)  

        #----> ppeg
        h = self.pos_layer(h, _H, _W) 

        #----> second translayer
        h = self.layer2(h)  

        #----> cls_token
        h = self.norm(h)[:, 0]

        #----> predict
        logits = self._fc2(h) 

        return logits