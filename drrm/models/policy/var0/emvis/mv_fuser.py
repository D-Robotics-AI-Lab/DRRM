import torch
from torch import Tensor, nn
from .layers import get_proj_layer, AttnFuser, dim_pooling

class MultiViewFuser(nn.Module):
    def __init__(
        self, 
        dim_in: int,
        dim_out: int = None,
        hidden: int = None,
        depth: int = 2,
        mlp_ratio = 4.0,
        ffn_layer_num = 1,
        drop_p = 0.,
        ** kwargs
    ) -> None:
        super().__init__()
        # dim_cat = dim_in * view_num
        if dim_out == None: dim_out = dim_in
        if hidden is None: hidden = dim_out

        # 2D 3D Fusion
        fuser_config = {
            'input_dim_list': [('FV', hidden), ('OV', hidden)],
            'dim_out': hidden,
            'mlp_ratio': mlp_ratio,
            'ffn_layer_num': ffn_layer_num,
            'drop_p': drop_p,
            ** kwargs
        }
        # self.modality_fuser = DimFuser(**fuser_config) if fuse_2d else nn.Identity
        self.fuser = nn.ModuleList([
            AttnFuser(**fuser_config) for _ in range(depth)
        ])

    def forward(self, tokens: Tensor, pos: Tensor, camera_tokens: Tensor = None) -> Tensor:
        BS, V, P, D = tokens.shape

        x = torch.ones(BS, P, D, device=tokens.device) # [B, P, D]
        pos_x = pos[:,0,...].view(BS, -1, 2) # [B, P, 2]

        y = tokens # [B, V, P, D]
        pos_y = pos # [B, V, P, 2]
        if camera_tokens is not None:
            camera_tokens = dim_pooling(camera_tokens, D) # [B, V, D]
            y = torch.cat((y, camera_tokens[...,None,:]), dim=-2) # [B, V, P+1, D]
            pos_y = pos_y + 1
            pos_special = torch.zeros(BS, V, 1, 2, device=pos_y.device).to(pos.dtype) # [B, V, 1, 2]
            pos_y = torch.cat([pos_special, pos_y], dim=-2) # [B, V, P+1, 2]
        y = y.view(BS, -1, D) # [B, V*(P+1), D]
        pos_y = pos_y.view(BS, -1, 2) # [B, V*(P+1), 2]

        for i, block in enumerate(self.fuser):
            x = block(x, y, pos_x, pos_y)
        return x.view(BS, 1, P, D)
    
if __name__ == "__main__":
    mvf = MultiViewFuser(1024,3)
    x = torch.rand((128,3,524,1024))
    x = mvf(x)
    print(x)