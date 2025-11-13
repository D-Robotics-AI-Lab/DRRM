import torch
from torch import Tensor, nn
from .layers import get_proj_layer, AttnFuser

class MultiViewFuser(nn.Module):
    def __init__(
        self, 
        dim_in: int,
        dim_out: int = None,
        mlp_ratio = 4.0,
        ffn_layer_num = 1,
        drop_p = 0.,
        ** kwargs
    ) -> None:
        super().__init__()
        # dim_cat = dim_in * view_num
        if dim_out == None: dim_out = dim_in
        # self.fuser = get_proj_layer(
        #     __target__="MLP",
        #     dim_in = dim_cat,
        #     dim_out = dim_out,
        #     mlp_ratio = mlp_ratio,
        #     layer_num = ffn_layer_num,
        #     drop_p = drop_p
        # )

        # 2D 3D Fusion
        fuser_config = {
            'input_dim_list': [('FV', dim_out), ('OV', dim_out)],
            'dim_out': dim_out,
            'mlp_ratio': mlp_ratio,
            'ffn_layer_num': ffn_layer_num,
            'drop_p': drop_p,
            ** kwargs
        }
        # self.modality_fuser = DimFuser(**fuser_config) if fuse_2d else nn.Identity
        self.fuser = AttnFuser(**fuser_config)

    def forward(self, tokens: Tensor, pos: Tensor) -> Tensor:
        BS, V, P, D = tokens.shape
        assert V>1
        x = tokens[:,0:1,...].view(BS, -1, D)
        y = tokens[:,1:,...].view(BS, -1, D)
        pos_x = pos[:,0:1,...].view(BS, -1, 2)
        pos_y = pos[:,1:,...].view(BS, -1, 2)
        fused_tokens = self.fuser(x, y, pos_x, pos_y)
        return fused_tokens.view(BS, 1, P, D)
    
if __name__ == "__main__":
    mvf = MultiViewFuser(1024,3)
    x = torch.rand((128,3,524,1024))
    x = mvf(x)
    print(x)