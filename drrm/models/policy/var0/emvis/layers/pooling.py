

from torch import nn

def dim_pooling(x, dim_out):
    pool = nn.AdaptiveAvgPool1d(dim_out)
    return pool(x)