""" from https://github.com/jaywalnut310/glow-tts """

import math

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
from einops import rearrange

import matcha.utils as utils  # pylint: disable=consider-using-from-import
from matcha.utils.model import sequence_mask

log = utils.get_pylogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.conv_layers.append(torch.nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DurationPredictor(nn.Module):
    """
    phoneme 임베딩을 받아서 convolution을 통해 dim이 1인, scalar 값 예측
    """
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.
    """

    def __init__(self, d: int, base: int = 10_000):
        r"""
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        r"""
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        x = rearrange(x, "b h t d -> t b h d")

        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])

        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 가정: RotaryPositionalEmbeddings(d_rot) 가 같은 파일/모듈에 이미 정의되어 있음
# from .rope import RotaryPositionalEmbeddings

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = False,
        rope_ratio: float = 0.5,  # RoPE를 head dim의 앞부분에만 적용 (0~1)
    ):
        """
        Args:
            channels: 입력 채널 (= H * Dh)
            out_channels: 출력 채널
            n_heads: 헤드 수 H
            p_dropout: attention dropout prob
            proximal_bias: self-attn에서 토큰 거리가 가까울수록 가산 bias (fallback 경로 필요)
            proximal_init: q/k 초기 가중치 동일 복사 (self-attn 안정화용)
            rope_ratio: RoPE를 적용할 비율 (0~1). Dh * rope_ratio 는 반드시 짝수 되도록 조정.
        """
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias

        # --- Projections ---
        # Self-attn fast path: qkv를 한 번에 추출
        self.qkv_conv = nn.Conv1d(channels, channels * 3, kernel_size=1)

        # Cross-attn fast path: q는 x에서, kv는 c에서 한 번에 추출
        self.q_conv  = nn.Conv1d(channels, channels, kernel_size=1)
        self.kv_conv = nn.Conv1d(channels, channels * 2, kernel_size=1)

        # 출력 프로젝션
        self.o_conv = nn.Conv1d(channels, out_channels, kernel_size=1)

        # 초기화
        nn.init.xavier_uniform_(self.qkv_conv.weight)
        nn.init.xavier_uniform_(self.q_conv.weight)
        nn.init.xavier_uniform_(self.kv_conv.weight)
        nn.init.xavier_uniform_(self.o_conv.weight)

        if proximal_init:
            # q/k weight를 동일 복사 (self-attn 안정화를 원할 때 선택)
            with torch.no_grad():
                # qkv_conv에서 q,k,v 순서로 분할되어 있다고 가정
                q_w, k_w, v_w = self.qkv_conv.weight.chunk(3, dim=0)
                q_b, k_b, v_b = self.qkv_conv.bias.chunk(3, dim=0)
                k_w.copy_(q_w)
                k_b.copy_(q_b)

        # --- RoPE 설정 ---
        # head_dim * rope_ratio 를 짝수로 맞춤
        d_rope = int(self.head_dim * rope_ratio)
        if d_rope % 2 == 1:
            d_rope -= 1
        d_rope = max(d_rope, 0)
        self.d_rope = d_rope
        if d_rope > 0:
            self.query_rope = RotaryPositionalEmbeddings(d_rope)
            self.key_rope   = RotaryPositionalEmbeddings(d_rope)
        else:
            self.query_rope = None
            self.key_rope   = None

        self.drop = nn.Dropout(p_dropout)
        self.attn = None  # 마지막 fallback 경로의 attn 가중치 저장용 (필요 시)

    # SDPA에서 사용할 커널 설정 (가능하면 Flash / 메모리 효율 커널 사용)
    @torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    def _sdpa(self, q, k, v, attn_mask, dropout_p, is_causal=False):
        # attn_mask: (B,H,Tq,Tk) bool (True=keep)
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor):
        """
        q, k: (B,H,T,Dh)
        앞쪽 d_rope 차원에만 RoPE 적용, 나머지는 통과
        """
        if self.d_rope <= 0:
            return q, k
        q_rope, q_pass = q[..., : self.d_rope], q[..., self.d_rope :]
        k_rope, k_pass = k[..., : self.d_rope], k[..., self.d_rope :]

        q_rope = self.query_rope(q_rope)
        k_rope = self.key_rope(k_rope)

        q = torch.cat([q_rope, q_pass], dim=-1)
        k = torch.cat([k_rope, k_pass], dim=-1)
        return q, k

    @staticmethod
    def _expand_mask_for_sdpa(mask, B, H, Tq, Tk):
        """
        다양한 입력 마스크를 SDPA용 (B,H,Tq,Tk) bool mask(True=keep)로 확장.
        허용 형태:
          - (B,Tk)       : key padding mask
          - (B,Tq,Tk)    : full attn mask
          - (B,1,Tq,Tk)  : broadcast head 차원
          - (B,H,Tq,Tk)  : 이미 완전한 형태
        """
        if mask is None:
            return None

        if mask.dtype != torch.bool:
            mask = mask != 0  # 0/1, float 마스크를 bool로 변환

        if mask.ndim == 2:
            # (B,Tk) -> (B,1,1,Tk) -> (B,H,Tq,Tk)
            mask = mask.unsqueeze(1).unsqueeze(1).expand(B, H, Tq, Tk)
        elif mask.ndim == 3:
            # (B,Tq,Tk) -> (B,1,Tq,Tk) -> (B,H,Tq,Tk)
            mask = mask.unsqueeze(1).expand(B, H, Tq, Tk)
        elif mask.ndim == 4:
            if mask.size(1) == 1:
                mask = mask.expand(B, H, Tq, Tk)
            else:
                # (B,H,Tq,Tk) 가정
                assert mask.size(1) == H and mask.size(2) == Tq and mask.size(3) == Tk
        else:
            raise ValueError(f"Unsupported attn mask shape: {mask.shape}")

        return mask

    @staticmethod
    def _attention_bias_proximal(length: int):
        # (1,1,T,T) 텐서
        r = torch.arange(length, dtype=torch.float32)
        diff = r[None, :] - r[:, None]
        bias = -torch.log1p(diff.abs())
        return bias[None, None, :, :]

    def forward(
        self,
        x: torch.Tensor,             # (B, C, Tq)
        c: torch.Tensor,             # (B, C, Tk)  (self-attn이면 x==c, Tq==Tk)
        attn_mask: torch.Tensor = None,  # 다양한 형태 허용 -> bool (B,H,Tq,Tk)로 확장
        is_causal: bool = False,
        return_attn: bool = False,   # True면 attn 가중치 반환 필요 → fallback 경로 사용
    ):
        B, C, Tq = x.shape
        _, _, Tk = c.shape
        H, Dh = self.n_heads, self.head_dim

        # --- Q/K/V 산출: self vs cross 분기 ---
        if x.data_ptr() == c.data_ptr() and Tq == Tk:
            # self-attention fast path: qkv fused
            qkv = self.qkv_conv(x)  # (B, 3C, T)
            q, k, v = qkv.split(self.channels, dim=1)
        else:
            # cross-attention fast path: q from x, kv from c
            q = self.q_conv(x)            # (B, C, Tq)
            kv = self.kv_conv(c)          # (B, 2C, Tk)
            k, v = kv.split(self.channels, dim=1)

        # (B,C,T) -> (B,H,T,Dh)
        q = rearrange(q, "b (h d) t -> b h t d", h=H)
        k = rearrange(k, "b (h d) t -> b h t d", h=H)
        v = rearrange(v, "b (h d) t -> b h t d", h=H)

        # RoPE
        q, k = self._apply_rope(q, k)

        # SDPA 마스크 준비
        sdpa_mask = self._expand_mask_for_sdpa(attn_mask, B, H, Tq, Tk)

        # --- Fast path (SDPA) vs Fallback 분기 ---
        use_sdpa = (not self.proximal_bias) and (not return_attn)

        if use_sdpa:
            # training일 때만 dropout 적용
            dropout_p = self.p_dropout if self.training else 0.0
            out = self._sdpa(q, k, v, sdpa_mask, dropout_p, is_causal=is_causal)  # (B,H,Tq,Dh)
            p_attn = None  # SDPA는 가중치 반환 X
        else:
            # Fallback: 수학 경로 (proximal bias 또는 attn weight 필요 시)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,Tq,Tk)

            if self.proximal_bias:
                # self-attention에서만 의미 있음
                if Tq != Tk:
                    raise ValueError("Proximal bias is only valid for self-attention (Tq must equal Tk).")
                bias = self._attention_bias_proximal(Tq).to(scores.device, scores.dtype)  # (1,1,T,T)
                scores = scores + bias

            if sdpa_mask is not None:
                # bool mask: False 위치에 큰 음수
                scores = scores.masked_fill(~sdpa_mask, -1e9)

            p_attn = torch.softmax(scores, dim=-1)
            p_attn = F.dropout(p_attn, p=self.p_dropout, training=self.training)
            out = torch.matmul(p_attn, v)  # (B,H,Tq,Dh)

        # (B,H,Tq,Dh) -> (B,C,Tq)
        out = rearrange(out, "b h t d -> b (h d) t")

        # 출력 프로젝션
        out = self.o_conv(out)  # (B, out_channels, Tq)

        # 필요 시 마지막 attn 저장(디버깅/로깅)
        self.attn = p_attn  # (B,H,Tq,Tk) or None

        return out
        # return out, p_attn  # p_attn=None일 수 있음

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = torch.nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Encoder(nn.Module):
    """
    SelfAttention + norm -> FeedForward + norm
    Feedforward is convoluion. Transformer 보다는 Unet
    """
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_type,
        encoder_params,
        duration_predictor_params,
        n_vocab,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = torch.nn.Embedding(n_vocab, self.n_channels)
        torch.nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)

        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x

        self.encoder = Encoder(
            encoder_params.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )

        # 최종 인코더 출력을 1×1 Conv로 n_feats 차원으로 사영.
        self.proj_m = torch.nn.Conv1d(self.n_channels + (spk_emb_dim if n_spks > 1 else 0), self.n_feats, 1)
        self.proj_w = DurationPredictor(
            self.n_channels + (spk_emb_dim if n_spks > 1 else 0),
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )

    def forward(self, x, x_lengths, spks=None):
        """Run forward pass to the transformer based encoder and duration predictor

        Args:
            x (torch.Tensor): text input
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): text input lengths
                shape: (batch_size,)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size,)

        Returns:
            mu (torch.Tensor): average output of the encoder
                shape: (batch_size, n_feats, max_text_length)
            logw (torch.Tensor): log duration predicted by the duration predictor
                shape: (batch_size, 1, max_text_length)
            x_mask (torch.Tensor): mask for the text input
                shape: (batch_size, 1, max_text_length)
        """
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.prenet(x, x_mask)
        if self.n_spks > 1:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask

        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask
