"""
2D Sinusoidal Positional Encoding for ECG-JEPA.
원본 pos_encoding.py에서 가져온 코드 (변경 없음).

Grid shape: (n_leads, n_patches)  →  embed_dim
"""

import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Args:
        embed_dim : total embedding dimension (must be even)
        grid_h    : number of rows    (= n_leads,   e.g. 12)
        grid_w    : number of columns (= n_patches, e.g. 50)

    Returns:
        pos_embed : np.ndarray  shape (grid_h * grid_w, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    grid_h_arr = np.arange(grid_h, dtype=np.float32)
    grid_w_arr = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_arr, grid_h_arr)   # (2, grid_h, grid_w)
    grid = np.stack(grid, axis=0)                # (2, grid_h, grid_w)
    grid = grid.reshape([2, 1, grid_h, grid_w])  # (2, 1, H, W)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed   # (H*W, embed_dim)


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)                        # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Args:
        embed_dim : D (must be even)
        pos       : (M,) or any shape of positions
    Returns:
        (M, D) array
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)   # (D/2,)

    pos = pos.reshape(-1)            # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)