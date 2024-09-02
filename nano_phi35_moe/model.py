import torch
from torch import nn

from nano_phi35_moe.config import Config
from nano_phi35_moe.components.layer import DecoderLayer


class MOEModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # save values that are used in the forward method
        self.sliding_window_size = config.sliding_window_size
        # Model Layers
        # padding_idx is used to set the embeddings of padding tokens to zeros
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) 
        self.decoder_layers = nn.ModuleList([DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)

        
    def forward(self, input_ids: torch.Tensor):
        """

        Args:
            input_ids (torch.Tensor): lists of token ids
        """
        # Step 1: convert token ids to embeddings
        batch_size, sequence_length = input_ids.size()
        token_embeddings: torch.Tensor = self.embeddings(input_ids) # shape: [batch_size, sequence_length, hidden_size]
         
        # Step 2: position embeddings
        position_ids = torch.arange(0, sequence_length, dtype=torch.long, device=input_ids.device).view(-1, sequence_length) # shape: [1, sequence_length]

        # Step 3: prepare attention mask
        # Our goal is to mask the tokens that are not supposed to be attended to.
        # It'd be easier to understand with this example:
        # Let's say we have sequence_length = 3
        # mask = [[-inf, -inf, -inf],
        #         [-inf, -inf, -inf],
        #         [-inf, -inf, -inf]]
        mask = torch.full((sequence_length, sequence_length), torch.finfo(token_embeddings.dtype).min, device=input_ids.device)
        # mask_condition = [0, 1, 2]
        mask_condition = torch.arange(0, sequence_length, device=input_ids.device)
        # (mask_condition + 1).view(sequence_length, 1) = [[1], [2], [3]]
        # After broadcasting, mask_condition becomes:
        # [[0, 1, 2],
        #  [0, 1, 2],
        #  [0, 1, 2]]
        # (mask_condition + 1).view(sequence_length, 1) becomes:
        # [[1, 1, 1],
        #  [2, 2, 2],
        #  [3, 3, 3]]
        # Thus the resulting Tensor is:
        # [[True, False, False],
        #  [True, True, False],
        #  [True, True, True]]
        # So the mask becomes:
        # [[0, -inf, -inf],
        #  [0, 0, -inf],
        #  [0, 0, 0]]
        mask.masked_fill_(mask_condition < (mask_condition + 1).view(sequence_length, 1), 0)
        
        mask.to(token_embeddings.dtype)
        
        context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=(-self.sliding_window_size - 1))
        mask.masked_fill_(context_mask, torch.finfo(token_embeddings.dtype).min)
        
        mask = mask[None, None, :, :].expand(batch_size, 1, sequence_length, sequence_length)
        
        hidden_states = token_embeddings
        for decoder_layer in self.decoder_layers:
            layer_output = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0]
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
        
