from collections import OrderedDict
from typing import Optional
from torch import Tensor, nn
from . import ActFunc

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = ActFunc()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def get_proj_layer(
    __target__: str,
    dim_in: int, 
    dim_out: int, 
    mlp_ratio: float = 1.0,
    layer_num: int = 2,
    drop_p: float = 0.
):
    if __target__ == "MLP":
        layer_list = []
        dim_mid = int(dim_in*mlp_ratio)
        for i in range(layer_num):
            if i == 0: # First
                layer_list += [(f'fc_{i}', nn.Linear(dim_in, dim_mid))]
            else: # Middel
                layer_list += [(f'fc_{i}', nn.Linear(dim_mid, dim_mid))]
            layer_list += [(f'act_{i}', ActFunc())]
            layer_list += [(f'drop_{i}', nn.Dropout(drop_p))]
        if layer_num != 0:
            layer_list += [(f'fc_{i+1}', nn.Linear(dim_mid, dim_out))]
        return nn.Sequential(
            OrderedDict(layer_list)
        )
    elif __target__ == "Nonlinear":
        return nn.Sequential(
            OrderedDict([
                (f'fc', nn.Linear(in_features = dim_in, out_features = dim_out)),
                (f'norm', nn.LayerNorm(dim_out)),
                (f'act', ActFunc()),
            ])
        )
    elif __target__ == "Linear":
        return nn.Sequential(
            OrderedDict([
                (f'fc', nn.Linear(in_features = dim_in, out_features = dim_out)),
                (f'norm', nn.LayerNorm(dim_out))
            ])
        )
    else:
        raise ValueError