from torch import Tensor, nn

class MyLayerNorm(nn.Module):
    def __init__(
        self,
        dim: int
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute((0,2,3,1))
        x = self.norm(x)
        x = x.permute((0,3,1,2))
        return x
    
class MyGroupNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        group: int,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute((1,0,2,3))
        x = self.norm(x)
        x = x.permute((1,0,2,3))
        return x