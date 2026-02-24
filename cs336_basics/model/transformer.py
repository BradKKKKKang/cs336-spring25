import torch
import torch.nn as nn
import cs336_basics.model.modules as modules
from cs336_basics.model.modules import RMSNorm, SwiGLU as FFN, multihead_self_attention

class transformer_block(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        d_ff, 
        max_seq_len, 
        theta, 
        device=None, 
        dtype=None,
        use_rope=False,
        ):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = multihead_self_attention(
            d_model,
            num_heads,
            use_rope,
            theta=theta
        )
        
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        self.ffn=FFN(d_model, d_ff)

        if device is not None and dtype is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, in_feature: torch.Tensor)->torch.Tensor:
        x = in_feature
        # Multihead_self_attention
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class transformer_lm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model, 
        num_heads, 
        d_ff, 
        theta, 
    ):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.num_layers=num_layers
        self.token_embedding = modules.Embedding(
            vocab_size, 
            d_model)
        self.layers = nn.ModuleList(
            [
                transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    use_rope=True
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = modules.RMSNorm(d_model)
        self.out_embedding = modules.Linear(d_model, vocab_size)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x=self.out_norm(x)
        x=self.out_embedding(x)
        return x