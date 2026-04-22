"""
models/ecg_jepa/model.py
─────────────────────────────────────────────────────────────────────────────
ECG-JEPA model (원본 ecg_jepa.py 구조를 그대로 보존, c=12 기본값으로만 변경).

원본: https://github.com/sehunfromdaegu/ECG_JEPA/blob/master/ecg_jepa.py
변경점:
  - c=12 (12-lead HEEDB 전용)
  - base_model.BaseModel 상속 추가 (레지스트리 연동)
  - 코드 정리 (기능 변경 없음)
"""

import copy
import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath

from .pos_encoding import get_2d_sincos_pos_embed
from ..base_model import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 빌딩 블록
# ─────────────────────────────────────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp  = Mlp(in_features=dim,
                        hidden_features=int(dim * mlp_ratio),
                        act_layer=act_layer, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, attention_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), attention_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=(drop_path_rate[i]
                             if isinstance(drop_path_rate, list)
                             else drop_path_rate),
                  norm_layer=norm_layer)
            for i in range(depth)
        ])

    def forward(self, x, pos, attention_mask=None):
        for block in self.blocks:
            x = block(x + pos, attention_mask)
        return x


class PredictorBlock(nn.Module):
    def __init__(self, predictor_embed_dim=192, depth=4, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=(drop_path_rate[i]
                             if isinstance(drop_path_rate, list)
                             else drop_path_rate))
            for i in range(depth)
        ])

    def forward(self, x, pos, attention_mask=None):
        for block in self.blocks:
            x = block(x + pos, attention_mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# MaskTransformer  (student / teacher encoder)
# ─────────────────────────────────────────────────────────────────────────────

class MaskTransformer(nn.Module):
    """
    CroPA (Cross-Pattern Attention) 적용 인코더.
    원본 코드 그대로, c=12 기본값.
    """

    def __init__(self, embed_dim=768, depth=12, num_heads=16,
                 mlp_ratio=4.0, qkv_bias=False, qk_scale=None,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, init_std=0.02,
                 mask_scale=(0.175, 0.225), mask_type='block',
                 pos_type='sincos', c=12, p=50, t=50,
                 leads=None):
        super().__init__()

        self.c = c
        self.p = p
        self.t = t
        self.embed_dim  = embed_dim
        self.mask_scale = mask_scale
        self.mask_type  = mask_type
        self.init_std   = init_std
        self.leads      = leads if leads is not None else list(range(c))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.encoder_blocks = EncoderBlock(
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr, norm_layer=norm_layer)

        self.norm = nn.LayerNorm(embed_dim)
        self.W_P  = nn.Linear(t, embed_dim)

        # Positional embedding (c × p, embed_dim)
        if pos_type == 'learnable':
            pos_embed = torch.empty((c * p, embed_dim))
            nn.init.uniform_(pos_embed, -0.02, 0.02)
            self.pos_embed = nn.Parameter(pos_embed, requires_grad=True)
        else:  # sincos
            self.pos_embed = nn.Parameter(
                torch.zeros(c * p, embed_dim), requires_grad=False)
            pe = get_2d_sincos_pos_embed(embed_dim, c, p)
            self.pos_embed.data.copy_(torch.from_numpy(pe).float())

        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.encoder_blocks.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data,   layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # ── 마스킹 ───────────────────────────────────────────────────────────────

    def _make_rand_mask(self, mask_scale):
        mask = torch.zeros(self.p, dtype=torch.bool)
        ratio = mask_scale[0] + (mask_scale[1] - mask_scale[0]) * torch.rand(1).item()
        n     = int(self.p * ratio)
        mask[torch.randperm(self.p)[:n]] = True
        return mask

    def _make_block_mask(self, mask_scale):
        mask = torch.zeros(self.p, dtype=torch.bool)
        for _ in range(4):
            ratio = mask_scale[0] + (mask_scale[1] - mask_scale[0]) * torch.rand(1).item()
            n     = int(self.p * ratio)
            start = torch.randperm(self.p - n)[0]
            mask[start:start + n] = True
        return mask

    # ── CroPA attention mask ─────────────────────────────────────────────────

    def _cross_attention_mask(self):
        size = self.c * self.p
        row_mask = torch.zeros((size, size))
        for i in range(self.c):
            row_mask[i*self.p:(i+1)*self.p, i*self.p:(i+1)*self.p] = 1
        col_mask = torch.zeros((size, size))
        for i in range(self.p):
            col_mask[i::self.p, i::self.p] = 1
        combined = (row_mask + col_mask).clamp(max=1)
        return combined

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x, mask=None):
        """
        x    : (B, c, p, t)
        mask : (p,) bool — 외부에서 주입하면 해당 마스크 사용
        """
        bs, c, p, t = x.shape
        assert c == self.c and p == self.p and t == self.t

        x = x.reshape(bs, c * p, t)
        pos_embed = self.pos_embed.unsqueeze(0).expand(bs, -1, -1)

        if mask is None:
            if self.mask_type == 'random':
                mask_idx = self._make_rand_mask(self.mask_scale)
            else:
                mask_idx = self._make_block_mask(self.mask_scale)
        else:
            mask_idx = mask

        vis_idx = (~mask_idx).repeat(c)

        x = self.W_P(x)   # (B, c*p, embed_dim)

        attn_mask = self._cross_attention_mask().to(x.device)

        if mask is not None:
            x         = x[:, vis_idx]
            pos_embed = pos_embed[:, vis_idx]
            attn_mask = attn_mask[vis_idx][:, vis_idx]

        x = self.encoder_blocks(x, pos_embed, attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x, mask_idx

    # ── 추론용 representation 추출 (downstream task) ─────────────────────────

    def representation(self, x):
        """
        x : (B, L, 2500)  — L은 self.c보다 작을 수 있음 (reduced-lead)
        반환: (B, embed_dim)  global avg pool
        """
        assert x.dim() == 3
        B, L, _ = x.shape
        x = x.reshape(B, -1, self.t)           # (B, L*p, t)
        x = self.W_P(x)                         # (B, L*p, embed_dim)

        pos_embed  = self.pos_embed
        attn_mask  = self._cross_attention_mask().to(x.device)

        # L < c 이면 해당 leads 만 슬라이싱
        if L < self.c:
            lead_idx   = list(range(L))
            rows = torch.cat([torch.arange(i*self.p, (i+1)*self.p)
                               for i in lead_idx])
            pos_embed = pos_embed[rows]
            attn_mask = attn_mask[rows][:, rows]

        pos_embed = pos_embed.unsqueeze(0)

        x = self.encoder_blocks(x, pos_embed, attn_mask)
        if self.norm:
            x = self.norm(x)
        return x.mean(dim=1)  # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# MaskTransformerPredictor
# ─────────────────────────────────────────────────────────────────────────────

class MaskTransformerPredictor(nn.Module):
    """Predictor: 각 리드 독립적으로 마스킹된 위치 예측."""

    def __init__(self, embed_dim=768, predictor_embed_dim=384,
                 depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 init_std=0.02, pos_type='sincos',
                 c=12, p=50, t=50):
        super().__init__()

        self.c = c
        self.p = p
        self.t = t

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.embed_dim        = embed_dim
        self.mask_token       = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # 1D positional embedding (predictor는 단일 리드 처리)
        if pos_type == 'learnable':
            pe = torch.empty((p, predictor_embed_dim))
            nn.init.uniform_(pe, -0.02, 0.02)
            self.pos_embed = nn.Parameter(pe, requires_grad=True)
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(p, predictor_embed_dim), requires_grad=False)
            pe = get_2d_sincos_pos_embed(predictor_embed_dim, 1, p)
            self.pos_embed.data.copy_(torch.from_numpy(pe).float())

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = PredictorBlock(
            predictor_embed_dim=predictor_embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr)

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        trunc_normal_(self.mask_token, std=init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.predictor_blocks.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data,   layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        """
        x    : (B, c*(p-n_mask), embed_dim)  — 컨텍스트 인코딩
        mask : (p,) bool
        """
        num_mask = mask.sum().item()
        x        = self.predictor_embed(x)
        bs, _, pred_dim = x.shape

        # 리드별로 분리
        x = x.reshape(bs * self.c, -1, pred_dim)   # (B*c, p_vis, pred_dim)

        mask_tokens = self.mask_token.expand(x.size(0), int(num_mask), pred_dim)
        x = torch.cat([x, mask_tokens], dim=1)      # (B*c, p, pred_dim)

        # 위치 재정렬
        vis_idx  = (~mask).nonzero(as_tuple=True)[0]
        mask_idx = mask.nonzero(as_tuple=True)[0]
        idx      = torch.cat((vis_idx, mask_idx))
        pos      = self.pos_embed[idx].unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.predictor_blocks(x, pos)
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)

        x = x.reshape(bs, -1, self.embed_dim)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# ECGJepa  (최상위 모델)
# ─────────────────────────────────────────────────────────────────────────────

class ECGJepa(BaseModel):
    """
    ECG-JEPA: student-teacher JEPA for 12-lead ECG.

    원본 ecg_jepa 클래스와 동일한 구조, c=12 기본값.
    BaseModel 상속으로 모델 레지스트리에 자동 등록.
    """

    # 레지스트리 키
    model_name = "ecg_jepa"

    def __init__(
        self,
        # encoder
        encoder_embed_dim   = 768,
        encoder_depth       = 12,
        encoder_num_heads   = 16,
        # predictor
        predictor_embed_dim = 384,
        predictor_depth     = 6,
        predictor_num_heads = 12,
        # shared
        mlp_ratio     = 4.0,
        qkv_bias      = False,
        qk_scale      = None,
        drop_rate     = 0.0,
        attn_drop_rate= 0.0,
        drop_path_rate= 0.1,
        norm_layer    = nn.LayerNorm,
        init_std      = 0.02,
        pos_type      = 'sincos',
        # ECG 구조
        mask_type     = 'block',
        mask_scale    = (0.175, 0.225),
        c             = 12,    # ← 12 leads (HEEDB 전체)
        p             = 50,    # patches per lead
        t             = 50,    # timepoints per patch  (50*50=2500)
        leads         = None,
    ):
        super().__init__()

        self.c = c
        self.p = p
        self.t = t

        self.encoder = MaskTransformer(
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_std=init_std, mask_scale=mask_scale,
            pos_type=pos_type, mask_type=mask_type,
            c=c, p=p, t=t, leads=leads,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = MaskTransformerPredictor(
            embed_dim=encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth, num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_std=init_std, pos_type=pos_type,
            c=c, p=p, t=t,
        )

        self.loss_fn = nn.SmoothL1Loss()

    # ── Pretraining forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, c, T)   T = p * t = 2500
        반환: scalar loss
        """
        B, c, T = x.shape
        assert T == self.p * self.t, f"Expected T={self.p*self.t}, got {T}"
        x = x.reshape(B, c, self.p, self.t)   # (B, c, p, t)

        # Teacher (no grad)
        with torch.no_grad():
            h, mask = self.target_encoder(x)                    # (B, c*p, D)
            h = torch.nn.functional.layer_norm(h, (h.size(-1),))
            h = h.reshape(B, self.c, self.p, -1)                # (B, c, p, D)
            masked_h = h[:, :, mask, :].reshape(B, -1, h.size(-1))

        # Student encoder
        z_ctx, _ = self.encoder(x, mask)                        # (B, c*(p-n), D)

        # Predictor
        z = self.predictor(z_ctx, mask)                         # (B, c*p, D)

        # Masked position 슬라이싱
        num_mask = mask.sum()
        z_pred = z.reshape(B, self.c, self.p, -1)[:, :, -num_mask:, :]
        z_pred = z_pred.reshape(B, -1, z.size(-1))

        return self.loss_fn(z_pred, masked_h)

    # ── Downstream representation 추출 ───────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, L, 2500)   — L ≤ c
        반환: (B, embed_dim)
        """
        return self.encoder.representation(x)