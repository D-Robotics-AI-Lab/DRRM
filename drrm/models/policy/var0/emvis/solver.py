import torch
from torch import nn
from .adapter import ResNetAdapter
from timm.models.vision_transformer import Mlp, RmsNorm

class FinalLayer(nn.Module):
    """
    The final layer of RDT.
    """
    def __init__(self, hidden_size, out_channels, residual):
        super().__init__()
        self.residual = residual
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels, 
            act_layer=approx_gelu, drop=0
        )
        self.pool = nn.AdaptiveAvgPool1d(out_channels)

    def forward(self, x):
        if self.residual:
            res = self.pool(x)
            x = self.norm_final(x)
            x = self.ffn_final(x)
            return x+res
        else:
            x = self.norm_final(x)
            x = self.ffn_final(x)
            return x

class Solver(nn.Module):
    """
    Class for var0 Solvers.
    """
    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        shape_in: tuple,
        depth=1,
        dtype=torch.bfloat16,
        residual=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.shape_in = shape_in
        self.dtype = dtype

        # self.blocks = ResNetAdapter(
        #     conv_num=depth,
        #     dim_in=hidden_size,
        #     dim_out=hidden_size,
        #     shape_out=(1,1),
        #     final_project=False
        # )
        self.final_layer = FinalLayer(
            hidden_size = hidden_size, 
            out_channels = action_dim,
            residual = residual
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(self, x):
        """
        x: (B * T, P, D), state + action token sequence, T = horizon,
            dimension D is assumed to be the same as the hidden size,
            P = h * w is the number of patches.
        """
        h, w = self.shape_in
        x_pos = torch.stack(
            (torch.meshgrid(torch.arange(h), 
                            torch.arange(w))), 
            dim=-1
        ).view(1,-1,2).to(x.device)
        # Forward pass
        # x = self.blocks(x, x_pos) # (B*T, P, D_hidden)
        # Inject the language condition at the final layer
        x = self.final_layer(x) # (B*T, P, D_action)
        return x