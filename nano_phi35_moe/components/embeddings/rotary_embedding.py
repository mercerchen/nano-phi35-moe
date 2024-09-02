import torch
import torch.nn as nn
from torch import Tensor

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: str = 'cpu',
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
    def _set_cost_sin_cache(self, seq_len, device, dtype):
        pass
    
    def foward(self, x: Tensor):
        pass