#!/usr/bin/env python
"""dx_umap_paper.py — paper-ready dx UMAP figure (2 row × 6 col + side legend).

Layout:
    [모델0 panels × 3] [inter] [모델1 panels × 3]   |   [legend]
    [모델2 panels × 3] [inter] [모델3 panels × 3]   |

- font: Times New Roman, 7pt
- 전체 width 약 3.0 inch (논문 페이지 1/3)
- panel 안에 글자 X, legend 만 figure 옆에
- 저장: results/<YYYYMMDD_HHMMSS>/dx_umap_compat12.{pdf,svg,png}

Run (hbkim env):
    cd /home/irteam/local-node-d/tykim/visuallize
    python notebooks/dx_umap_paper.py
"""
import sys, os, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Times New Roman 등록 (mscorefonts conda 패키지 → /opt/conda/envs/.../fonts/) ──
import matplotlib
import matplotlib.font_manager as fm

def _ensure_times_new_roman():
    """matplotlib 에 Times New Roman 등록. mscorefonts (conda) 가 깔려 있으면 사용,
    없으면 candidate dirs 에서 times.ttf 찾아 addfont."""
    if any(f.name == 'Times New Roman' for f in fm.fontManager.ttflist):
        return True
    candidates = [
        Path(sys.prefix) / 'fonts' / 'times.ttf',
        Path('/home/irteam/local-node-d/_conda/envs/hbkim/fonts/times.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'),
        Path.home() / '.local/share/fonts/times.ttf',
    ]
    for p in candidates:
        if p.exists():
            fm.fontManager.addfont(str(p))
            bd = p.with_name('timesbd.ttf')
            if bd.exists():
                fm.fontManager.addfont(str(bd))
            return any(f.name == 'Times New Roman' for f in fm.fontManager.ttflist)
    return False

_has_tnr = _ensure_times_new_roman()
if not _has_tnr:
    print('[warn] Times New Roman 미발견 — DejaVu Serif 로 fallback')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.text as mtext

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': (['Times New Roman'] if _has_tnr else []) + ['DejaVu Serif'],
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# ── 데이터/모델 ────────────────────────────────────────
import scripts.umap_view as uv
from scripts.umap_view import quick_dx, dx_compatibility_table

DISPLAY_NAMES = {
    'CPC':         'ECG-CPC',
    'Ours-cb1024': 'MoRyECG(Ours)',
}

def make_figure(target_w_in=3.0,
                legend_w_in=0.95,
                legend_gap_in=0.04,
                intra_in=0.02,
                inter_in=0.04,
                vrow_in=0.04,
                margin_in=0.04,
                models=('CPC', 'ECG-FM', 'ECG-JEPA', 'Ours-cb1024'),
                balance_per_code=120,
                scatter_s=0.35,
                scatter_alpha=0.7):
    """target_w_in 안에 panel + legend 다 넣기. panel size 자동 계산.

    panel 이 작을수록 (target_w_in ↓) scatter_s 도 작아져야 점이 안 뭉개짐.
    balance_per_code 도 같이 줄이면 더 깨끗.
    """
    compat = dx_compatibility_table()
    compat_codes = compat[compat['compatible']]['icd'].tolist()

    n_age = 3
    models_per_row = 2
    n_models = len(models)
    tgt_rows = (n_models + models_per_row - 1) // models_per_row

    # panel width 역산: panels_total = 2*(3*p + 2*intra) + inter
    panels_total_w = target_w_in - legend_w_in - legend_gap_in - 2 * margin_in
    panel_in = (panels_total_w - 4 * intra_in - inter_in) / 6
    if panel_in <= 0.1:
        raise ValueError(f'target_w_in={target_w_in} 너무 작음 (panel_in={panel_in:.3f})')

    # quick_dx 호출 — Figure.legend / Figure.suptitle 은 호출 동안 캡처/no-op
    captured = {'handles': []}
    def _cap(self, *a, **kw):
        h = kw.get('handles') or (a[0] if a else None) or []
        captured['handles'] = list(h)
        return None
    _ol, _os = Figure.legend, Figure.suptitle
    Figure.legend = _cap
    Figure.suptitle = lambda self, *a, **kw: None
    try:
        fig, metrics = quick_dx(
            models=list(models),
            include_codes=compat_codes,
            age_split=18.0,
            balance_per_code=balance_per_code,
            show=False, show_metrics=False,
            figsize_per_cell=(panel_in, panel_in),
            s=scatter_s, alpha=scatter_alpha,
        )
    finally:
        Figure.legend, Figure.suptitle = _ol, _os

    raw_axes = list(fig.axes)
    group_w_in = n_age * panel_in + (n_age - 1) * intra_in

    # legend 높이 측정 (대략) → panel area 와 figure 높이 결정
    n_leg = len(captured['handles'])
    leg_line_h_in = 7 / 72 * 1.4  # 7pt + line spacing
    leg_h_in = max(0.5, n_leg * leg_line_h_in + 0.1)

    panels_h_in = tgt_rows * panel_in + (tgt_rows - 1) * vrow_in
    fig_h_in = max(panels_h_in, leg_h_in) + 2 * margin_in
    fig_w_in = (margin_in + 2 * group_w_in + inter_in
                + legend_gap_in + legend_w_in + margin_in)
    fig.set_size_inches(fig_w_in, fig_h_in)

    # panel 들을 figure 세로 가운데 정렬
    panels_y_top = (fig_h_in + panels_h_in) / 2
    for i, ax in enumerate(raw_axes):
        m, a = i // n_age, i % n_age
        nr = m // models_per_row
        m_in_row = m % models_per_row
        x_in = margin_in + m_in_row * (group_w_in + inter_in) + a * (panel_in + intra_in)
        y_top_in = panels_y_top - nr * (panel_in + vrow_in)
        y_in = y_top_in - panel_in
        ax.set_position([x_in / fig_w_in, y_in / fig_h_in,
                         panel_in / fig_w_in, panel_in / fig_h_in])

    # 텍스트 박멸
    for ax in raw_axes:
        ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(labelbottom=False, labelleft=False, labeltop=False, labelright=False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for t in list(ax.texts):
            t.remove()
    for t in list(fig.texts):
        t.remove()
    if getattr(fig, '_suptitle', None) is not None:
        try: fig._suptitle.remove()
        except Exception: pass
        fig._suptitle = None
    for artist in fig.findobj(match=mtext.Text):
        if artist.get_text():
            artist.set_text('')

    # legend — figure 오른쪽, 세로 가운데
    handles = captured['handles']
    labels = [h.get_label() for h in handles]
    legend_x_frac = (margin_in + 2 * group_w_in + inter_in + legend_gap_in) / fig_w_in
    fig.legend(handles=handles, labels=labels,
               loc='center left',
               bbox_to_anchor=(legend_x_frac, 0.5),
               ncol=1, fontsize=7, frameon=False,
               handlelength=0.9, handletextpad=0.4,
               borderaxespad=0, labelspacing=0.4)

    # rename returned metrics dict keys
    metrics = {DISPLAY_NAMES.get(k, k): v for k, v in metrics.items()}
    return fig, metrics, panel_in


def main():
    # 3.0" 는 12 panel + legend 에 너무 좁아서 다 뭉개짐.
    # 5.0" 로 잡으면 panel ≈ 0.65" → cluster 가 식별됨.
    fig, metrics, panel_in = make_figure(
        target_w_in=5.0,
        balance_per_code=150,
        scatter_s=0.6,
        scatter_alpha=0.6,
    )

    save_dir = PROJECT_ROOT / 'results' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    name = 'dx_umap_compat12'
    for ext, dpi in [('pdf', None), ('svg', None), ('png', 600)]:
        kw = {'dpi': dpi} if dpi else {}
        fig.savefig(save_dir / f'{name}.{ext}', **kw)

    fw, fh = fig.get_size_inches()
    print(f'[saved] {save_dir}/{name}.{{pdf,svg,png}}')
    print(f'figure: {fw:.2f}" × {fh:.2f}"   panel: {panel_in:.3f}"   font: '
          f'{"Times New Roman" if _has_tnr else "DejaVu Serif (fallback)"} 7pt')
    plt.close(fig)


if __name__ == '__main__':
    main()
