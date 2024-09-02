from dataclasses import dataclass

@dataclass
class Config:
    # tokenizer
    vocab_size: int = 50257
    pad_token_id: int = 0
    # model
    hidden_size: int = 768
    sliding_window_size: int = 256
    num_hidden_layers: int = 12
    rms_norm_eps: float = 1e-6
    lm_head_bias: bool = False