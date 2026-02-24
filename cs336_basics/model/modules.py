from locale import D_FMT
from math import e
from this import d

from numpy import dtype
from numpy.ma import masked_values
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.__init_weight()

    def __init_weight(self):
        mean = 0.0
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=mean, std=std, a = -3 * std, b =3 * std)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device=None,
        dtype: torch.dtype=None,
        ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embed_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._init_weight()
    def _init_weight(self):
        nn.init.trunc_normal_(self.embed_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    def forward(self, token_ids:torch.Tensor)->torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        return self.embed_weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device=None,
        dtype: torch.dtype=None,
        ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g_weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.g_weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        out = einsum(x/rms, self.g_weight, '... d, d ... -> ... d')
        return out.to(dtype=in_dtype) 

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        if dtype is None or torch.is_float_point(torch.empty(), dtype=dtype):
            dtype = torch.float32

        self.d_model = d_model
        self.d_ff = int(d_ff)
        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self):
        torch.nn.init.trunc_normal_(self.w1, mean=0.0, std=1.0, a=-3.0, b=3.0)
        torch.nn.init.trunc_normal_(self.w2, mean=0.0, std=1.0, a=-3.0, b=3.0)
        torch.nn.init.trunc_normal_(self.w3, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        a = einsum(self.w1, x, 'd_ff d_model, ... d_model-> ... d_ff')
        silu = a * torch.sigmoid(a)
        b = silu * einsum(self.w3, x, 'd_ff d_model, ... d_model-> ... d_ff')
        return einsum(self.w2, b, 'd_model d_ff, ... d_ff -> ... d_model')

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None =None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.freq_arrange = 1 / (self.theta ** (torch.arange(0, d_k, 2).to(dtype=torch.float) / self.d_k))
        self.register_buffer(name='inv_freq', tensor = self.freq_arrange, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)->torch.Tensor:
        B = x.size(0)
        S = x.size(-2)
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)
        else:
            if token_positions.dim() == 2:
                token_positions = token_positions[0]
        
        # dim: s, d_k/2
        theta_phase = einsum(token_positions, self.inv_freq, 's, d -> s d')
        # rotation: RoPE = x * cos + rotate(x) * sin
        cos = theta_phase.cos().repeat_interleave(2, dim=-1)[None, :, :]
        sin=theta_phase.sin().repeat_interleave(2,dim=-1)[None, :, :]
        rotate_x = self.rotate_tensor(x)
        return x * cos + rotate_x * sin

    def rotate_tensor(self, x: torch.Tensor)->torch.Tensor:
        # [x_2d, x_2d+1] -> [-x_2d+1, x_2d]
        x = rearrange(x, '... (s r) -> ... s r', r=2)
        x_even, x_odd = x.unbind(dim=-1)
        x = torch.stack((-x_odd, x_even), dim=-1)
        return rearrange(x, '... s r-> ... (s r)')

def softmax(x: torch.tensor, dim:int)->torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None
)->torch.Tensor:
    q_k_score = einsum(q, k, '... s_q d, ... s_k d -> ... s_q s_k') / (q.size(-1) ** 0.5)
    # add mask
    if mask != None:
        q_k_score = q_k_score.masked_fill(mask==False, -1e9)
    q_k_attention = softmax(q_k_score, dim=-1)
    return einsum(q_k_attention, v, '... s_q s_k, ... s_k d_v -> ... s_q d_v')

class multihead_self_attention(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads, 
        use_rope= False, 
        max_seq_len = 2048, 
        theta = 10000.0, 
        device=None, 
        dtype=None, 
        use_causal_mask=True
        ):
        super().__init__()

        self.pe = None
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        assert d_model % num_heads == 0, 'number of heads donen\' match d_model'

        self.d_k = d_model // num_heads
        
        self.w_q = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_k = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_v = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        
        if use_rope:
            self.pe = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

    def causal_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # linear transformation
        q_i = self.w_q(x)
        k_i = self.w_k(x)
        v_i = self.w_v(x)
        # multiheads
        q_i = rearrange(q_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        k_i = rearrange(k_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        v_i = rearrange(v_i, 'b s (n_h d_k) -> b n_h s d_k', n_h=self.num_heads)
        # position embedding
        if self.pe is not None:
            q_i = self.pe(q_i, token_positions)
            k_i = self.pe(k_i, token_positions)
        # use mask
        mask = None
        if self.use_causal_mask:
            mask = self.causal_mask(q_i.size(-2))
            mask = mask.to(device=q_i.device)
        # attention value calculate
        atten = scaled_dot_product_attention(q_i, k_i, v_i, mask)
        atten = rearrange(atten, 'b n_h s d_k -> b s (n_h d_k)', n_h=self.num_heads)
        out = self.w_o(atten)
        return out