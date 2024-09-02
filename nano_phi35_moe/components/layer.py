from torch import nn

from nano_phi35_moe.config import Config
from nano_phi35_moe.components.attentions.eager import EagerAttention

class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.self_attention = EagerAttention()
        self.block_sparse_moe = BlockSparseMoE()

    def forward(self, x):
        pass
