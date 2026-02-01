# ===== src/analysis.py =====
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================
# 基础工具
# =============================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def window_to_tensor(x_flat, window_size, feature_dim):
    """
    把 [1, W*F] 的扁平窗口还原成 [W, F]
    """
    assert x_flat.ndim == 2 and x_flat.shape[0] == 1, f"expect [1, W*F], got {x_flat.shape}"
    return x_flat.view(window_size, feature_dim).detach().cpu().numpy()

@torch.no_grad()
def get_attention_on_sample(model, x_flat):
    """
    若模型实现了 discriminate_with_attn，则返回 (score, attn_TT)；
    否则返回 (score, None)。兼容 [B,H,T,T] / [B,T,T] / [T,T]。
    """
    if hasattr(model, "discriminate_with_attn"):
        out, attn = model.discriminate_with_attn(x_flat)
        score = out.squeeze().detach().cpu().item()
        attn = attn.detach().cpu().numpy()
        if attn.ndim == 4:      # [B, H, T, T] -> 按 head 平均
            attn = attn[0].mean(axis=0)
        elif attn.ndim == 3:    # [B, T, T]
            attn = attn[0]
        elif attn.ndim == 2:    # [T, T]
            pass
        else:
            raise RuntimeError(f"Unexpected attn shape: {attn.shape}")
        return score, attn
    else:
        out = model.discriminate(x_flat)
        score = out.squeeze().detach().cpu().item()
        return score, None

def plot_attention_and_series(save_dir, title, series_WF, attn_TT=None, label_WF=None):
    """
    绘制：
      1) 注意力热图 (T×T)
      2) 多变量时间窗曲线 (W×F)
      3) 若提供标签，绘制时刻掩膜
    """
    W, F = series_WF.shape

    if attn_TT is not None:
        assert attn_TT.ndim == 2 and attn_TT.shape[0] == attn_TT.shape[1], "attention 必须是 [T,T]"
        plt.figure(figsize=(6,5))
        plt.imshow(attn_TT, aspect='auto', origin='lower')
        plt.xlabel('Key time'); plt.ylabel('Query time'); plt.title(f'Attention Map ({title})')
        plt.colorbar(); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{title}_attn.png'), dpi=200); plt.close()

    plt.figure(figsize=(10,4))
    for f in range(F):
        plt.plot(series_WF[:, f], alpha=0.8)
    plt.title(f'Sequence Window ({title})'); plt.xlabel('Time'); plt.ylabel('Value')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{title}_series.png'), dpi=200); plt.close()

    if label_WF is not None:
        mask_t = (label_WF.sum(axis=1) > 0).astype(int)
        plt.figure(figsize=(10,2))
        plt.imshow(mask_t[None, :], aspect='auto', cmap='Reds', interpolation='nearest')
        plt.yticks([]); plt.xlabel('Time'); plt.title(f'Anomaly Timeline ({title})')
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{title}_label_timeline.png'), dpi=200); plt.close()

def plot_overlay_small_multiples(
    ab_WF,                # (W, F) 异常窗口
    normal_bank_WF,       # (Nref, W, F) 参考正常库（已对齐窗口）
    saliency_F,           # (F,) 通道归因分数
    attn_alpha=None,      # (Q, K) 注意力权重；没有就传 None
    topk=6,               # 只展示 Top-K 通道
    band='std',           # 'std' 或 'iqr'
    save_path=None,       # 若给路径则保存图片
    title='Overlay per channel (Top-K saliency)',
    figsize=(10, 8)
):
    """
    画 Top-K 通道的小 multiples：
    - 粗线：异常窗口；细线：最近邻正常均值；阴影：μ±σ 或 IQR
    - 虚线：注意力峰值 key times（若提供 attn_alpha）
    """
    W, F = ab_WF.shape
    top_idx = np.argsort(np.asarray(saliency_F))[::-1][:topk]

    # 正常均值与带宽
    mu = normal_bank_WF.mean(axis=0)        # (W, F)
    if band == 'std':
        lo = mu - normal_bank_WF.std(axis=0)
        hi = mu + normal_bank_WF.std(axis=0)
        band_label = r"$\mu \pm \sigma$"
    else:
        q25 = np.percentile(normal_bank_WF, 25, axis=0)
        q75 = np.percentile(normal_bank_WF, 75, axis=0)
        lo, hi = q25, q75
        band_label = "IQR"

    # 注意力峰值 key times（用列均值取前2）
    peak_keys = []
    if attn_alpha is not None:
        col_mean = attn_alpha.mean(axis=0)           # (K,)
        peak_keys = np.argsort(col_mean)[::-1][:2]   # 取两个峰

    fig, axes = plt.subplots(len(top_idx), 1, sharex=True, figsize=figsize)
    if len(top_idx) == 1:
        axes = [axes]
    t = np.arange(W)

    for ax, f in zip(axes, top_idx):
        # 正常均值 + 带宽
        ax.plot(t, mu[:, f], lw=1.5, label='nearest-normal mean', alpha=0.95)
        ax.fill_between(t, lo[:, f], hi[:, f], alpha=0.15, label=band_label)
        # 异常
        ax.plot(t, ab_WF[:, f], lw=2.6, label='abnormal', zorder=3)
        # 注意力 key times
        for k in peak_keys:
            if 0 <= k < W:
                ax.axvline(k, ls='--', lw=1, alpha=0.6)
        ax.set_ylabel(f'feat {int(f)}')
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel('Time')
    # 只在最后一个子图放图例
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=3, frameon=True, loc='upper center')

    # 标题与布局
    fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def topk_attended_timesteps(attn_TT, k=5):
    col_importance = attn_TT.mean(axis=0)   # [T]
    idx = np.argsort(col_importance)[::-1][:k]
    return idx, col_importance[idx]

def grad_input_saliency(model, x_flat):
    """
    对判别器输出 y 对输入 x 的梯度，返回 |grad * x| 作为特征重要性（[1, W*F]）
    """
    x = x_flat.clone().detach().requires_grad_(True)
    y = model.discriminate(x)  # 标量分数（WGAN 判别器分数）
    y.sum().backward()
    sal = torch.abs(x.grad * x.detach()).detach().cpu().numpy()  # [1, W*F]
    return sal

# === NEW: 用注意力把逐时刻显著性聚合到“通道级” ===
def aggregate_feature_saliency(saliency_TF, attn_TT=None):
    """
    saliency_TF: [T, F] 逐时间步 |grad*x|
    attn_TT:     [T, T] 注意力(可选)。若无则均匀权重。
    return:      [F]    通道级加权归因(归一化到 sum=1)
    """
    T, F = saliency_TF.shape
    if attn_TT is None:
        w_t = np.ones(T, dtype=np.float32) / max(1, T)
    else:
        # 用“对 key 的全局权重”作为时间权重（列平均）
        w_t = attn_TT.mean(axis=0).astype(np.float64)  # [T]
        s = w_t.sum()
        w_t = (w_t / s) if s > 0 else np.ones(T, dtype=np.float64) / max(1, T)

    # 时间加权聚合到通道
    sal_F = (saliency_TF * w_t[:, None]).sum(axis=0)
    # L1 归一
    s = np.sum(np.abs(sal_F)) + 1e-12
    return (sal_F / s).astype(np.float64)

# === NEW: 让 Top-K 统计口径与上面一致（时间加权） ===
def weighted_topk_scores(x_TF, mu_F, sigma_F, attn_TT=None):
    """
    x_TF:    [T, F] 当前窗口原始数值(或已标准化)
    mu_F, sigma_F:  正常库均值/方差 [F]
    attn_TT: [T, T] 注意力(可选)
    return:  diff_F, z_F  —— 均已做 L1 归一，便于与 sal_F 比较/排序
    """
    T, F = x_TF.shape
    if attn_TT is None:
        w_t = np.ones(T, dtype=np.float64) / max(1, T)
    else:
        w_t = attn_TT.mean(axis=0).astype(np.float64)
        s = w_t.sum()
        w_t = (w_t / s) if s > 0 else np.ones(T, dtype=np.float64) / max(1, T)

    # 局部偏离：用一阶差分的绝对值并做时间加权（比“取最大”更稳）
    if T >= 2:
        dx_TF = np.abs(np.diff(x_TF, axis=0))
        w_diff = w_t[1:]
        w_diff = (w_diff / (w_diff.sum() + 1e-12))
        diff_F = (dx_TF * w_diff[:, None]).sum(axis=0)
    else:
        diff_F = np.zeros(F, dtype=np.float64)

    # 全局罕见：z-score 的绝对值并做时间加权
    z_TF = np.abs((x_TF - mu_F[None, :]) / (sigma_F[None, :] + 1e-6))
    z_F  = (z_TF * w_t[:, None]).sum(axis=0)

    # L1 归一到 sum=1，便于排序与对比
    diff_F = diff_F / (diff_F.sum() + 1e-12)
    z_F    = z_F / (z_F.sum() + 1e-12)
    return diff_F, z_F

# ==== CHANGED: 稳健归一的特征重要性（缓解维度量纲影响） ====
def _normalize_feat_importance_from_saliency(sal_WF, cap_pct=99.5):
    """
    输入: sal_WF [W, F]，为 |grad*x| 在窗口维度上的分布
    处理:
      1) 对每个特征在时间维做百分位截断，抑制极端峰值；
      2) 按时间求和得到每特征原始重要性；
      3) 做 L1 归一 (sum=1)，得到可比较的特征重要性。
    返回: feat_score_norm [F], feat_score_raw [F]
    """
    s = np.abs(sal_WF).astype(np.float64)            # [W, F]
    # per-feature percentile cap
    caps = np.percentile(s, cap_pct, axis=0)
    caps = np.maximum(caps, 1e-12)
    s = np.minimum(s, caps[None, :])
    feat_score_raw = s.sum(axis=0)                   # [F]
    denom = np.sum(feat_score_raw) + 1e-12
    feat_score_norm = feat_score_raw / denom
    return feat_score_norm, feat_score_raw

def plot_feature_saliency(save_dir, title, feat_score, feature_dim=None):
    plt.figure(figsize=(6,4))
    idx = np.arange(len(feat_score)) if feature_dim is None else np.arange(feature_dim)
    plt.bar(idx, feat_score)
    plt.xlabel('Feature index'); plt.ylabel('Importance'); plt.title(f'Feature Saliency ({title})')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{title}_feat_saliency.png'), dpi=200); plt.close()

# =============================
# 数据本体对比：异常 vs 正常参考
# =============================
def cosine_distances_np(x, Y):
    """
    x: [D] or [1,D]; Y: [N,D] -> 返回 [N] 余弦距离
    """
    x = x.reshape(1, -1)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    sim = (x @ Y.T) / (x_norm * Y_norm.T)
    return (1.0 - sim).ravel()

def fit_normal_reference(normal_windows_WF):
    """
    输入: 正常参考窗口集合，形状 [N_norm, W, F]
    输出: 每特征均值/标准差，以及参考相关矩阵
    """
    X = normal_windows_WF.reshape(-1, normal_windows_WF.shape[-1])  # [N_norm*W, F]
    mu = X.mean(axis=0)            # [F]
    sigma = X.std(axis=0) + 1e-6   # [F]
    try:
        corr = np.corrcoef(X, rowvar=False)
    except Exception:
        corr = np.eye(X.shape[1])
    return mu, sigma, corr

def window_stats_vs_normal(x_WF, mu_F, sigma_F, ref_corr_FF, normal_bank_WF=None, topk=10, save_dir='explain'):
    """
    对单个窗口 x 与正常参考做全面对比，返回统计摘要并出图
    """
    W, F = x_WF.shape

    # 1) 最大绝对z-score（逐特征，窗口内取max）
    z = (x_WF - mu_F[None, :]) / sigma_F[None, :]
    z_abs_max_F = np.abs(z).max(axis=0)  # [F]

    # 2) 差分突变（逐特征）
    if W >= 2:
        dx = np.diff(x_WF, axis=0)
        dx_abs_max_F = np.abs(dx).max(axis=0)
    else:
        dx_abs_max_F = np.zeros(F)

    # 3) 相关结构偏移
    corr_x = np.corrcoef(x_WF, rowvar=False) if W > 2 else np.eye(F)
    corr_shift = np.linalg.norm(corr_x - ref_corr_FF, ord='fro')

    # 4) 近邻正常窗口（可选）
    # 4) 近邻正常窗口（可选）
    nn_info = None
    if normal_bank_WF is not None and len(normal_bank_WF) > 0:
        # 近邻检索（与原来一致）
        x_flat = x_WF.reshape(-1)
        bank_flat = normal_bank_WF.reshape(len(normal_bank_WF), -1)
        d = cosine_distances_np(x_flat, bank_flat)
        nn_idx = np.argsort(d)[:min(5, len(d))]
        nn_bank = normal_bank_WF[nn_idx]
        nn_info = {"indices": nn_idx, "distances": d[nn_idx], "nn_examples": nn_bank}

        # === 新：Top-K 小 multiples（替代原来的“大杂烩叠加图”） ===
        overlay_grid_path = os.path.join(save_dir, "overlay_topk_grid.png")

        # 归因向量名兼容：你代码里可能叫 feat_saliency 或 saliency_F
        if 'feat_saliency' in locals():
            sal_vec = np.asarray(feat_saliency)
        elif 'saliency_F' in locals():
            sal_vec = np.asarray(saliency_F)
        else:
            # 兜底：用与最近邻均值的最大绝对差作为启发式 saliency，避免为空
            sal_vec = np.abs(x_WF - nn_bank.mean(axis=0)).max(axis=0)

        # 注意力矩阵（若没有就传 None）
        attn_mat = alpha_qk if 'alpha_qk' in locals() else None

        plot_overlay_small_multiples(
            ab_WF=x_WF,
            normal_bank_WF=nn_bank,
            saliency_F=sal_vec,
            attn_alpha=attn_mat,
            topk=6,            # 需要可改
            band='std',        # 或 'iqr'
            save_path=overlay_grid_path,
            title='Overlay per channel (Top-6 saliency)',
            figsize=(10, 8)
        )
        print(f"[explain] saved {overlay_grid_path}")


    # Top-K 图
    def bar_topk(values, title, fname):
        idx = np.argsort(-values)[:min(topk, len(values))]
        plt.figure(figsize=(8,3))
        plt.bar(range(len(idx)), values[idx])
        plt.xticks(range(len(idx)), idx)
        plt.title(title); plt.xlabel('Feature'); plt.ylabel('Score')
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, fname), dpi=200); plt.close()
        return idx, values[idx]

    topz_idx, topz_val = bar_topk(z_abs_max_F, 'Top-K Max|z-score| per feature', 'topk_zscore.png')
    topdx_idx, topdx_val = bar_topk(dx_abs_max_F, 'Top-K Max|diff| per feature', 'topk_diff.png')

    return {
        "z_abs_max_F": z_abs_max_F,
        "dx_abs_max_F": dx_abs_max_F,
        "corr_shift_Fro": float(corr_shift),
        "top_z_features": (topz_idx.tolist(), topz_val.tolist()),
        "top_diff_features": (topdx_idx.tolist(), topdx_val.tolist()),
        "nearest_normal": nn_info,
    }

def _build_label_WF(labels_np, idx, series_WF):
    """
    自适应构造 label_WF：优先 [W,F]，否则把 [F] 广播到时间维；不兼容则返回 None
    """
    label_WF = None
    try:
        lbl_slice = labels_np[idx:idx+1]
        W, F = series_WF.shape
        if lbl_slice.ndim == 2 and lbl_slice.shape[-1] == (W * F):
            label_WF = lbl_slice.reshape(W, F)
        elif lbl_slice.ndim == 2 and lbl_slice.shape[-1] == F:
            label_WF = np.tile(lbl_slice, (W, 1))
        elif lbl_slice.ndim == 1 and lbl_slice.size == F:
            label_WF = np.tile(lbl_slice[None, :], (W, 1))
        else:
            label_WF = None
    except Exception:
        label_WF = None
    return label_WF

# =============================
# 主入口一：常规单样本分析（异常窗 + 相邻正常窗）
# =============================
def run_attention_analysis(
    model,
    flat_dataset_tensor,      # [N, W*F]
    labels_np,                # [N, F] 或 [N]
    window_size,
    feature_dim,
    sample_index=None,
    out_dir='explain',
    train_flat_dataset_tensor=None,  # [Ntr, W*F]（可选）
    max_norm_bank=2000
):
    """
    生成：注意力热图、特征重要性、Top-K时刻、窗口 vs 正常参考的深度对比（含叠图与摘要）。
    - 若给出 train_flat_dataset_tensor，则以其为正常参考；否则用测试集中 label==0 的窗口作参考。
    - 同时对“异常窗 ab_idx”和“其相邻正常窗 norm_idx”各产出一套图/数值。
    """
    device = next(model.parameters()).device
    save_dir = _ensure_dir(out_dir)

    # --- 新增：长度对齐，避免 label 与数据长度不一致 ---
    N_data = int(flat_dataset_tensor.shape[0])
    N_label = int(labels_np.shape[0])
    N = min(N_data, N_label)
    if (N_data != N_label):
        print(f"[analysis] Aligning lengths: data={N_data}, labels={N_label} -> use N={N}")
    flat_dataset_tensor = flat_dataset_tensor[:N]
    labels_np = labels_np[:N]

    # 选择异常窗与相邻正常窗（基于对齐后的 labels_np）
    if sample_index is None:
        abnormal_idx_arr = np.where(_window_label(labels_np, feature_dim) > 0)[0]
        if len(abnormal_idx_arr) == 0:
            raise RuntimeError("测试集中未找到异常窗。")
        ab_idx = int(abnormal_idx_arr[0])
    else:
        ab_idx = int(sample_index)
    # 边界保护
    ab_idx = max(0, min(ab_idx, labels_np.shape[0]-1))
    norm_idx = max(0, ab_idx - 1)

    # 正常参考库：优先 train；否则用测试集里 label==0 的窗口
    if train_flat_dataset_tensor is not None:
        ref_flat = train_flat_dataset_tensor.detach().cpu().numpy()
    else:
        normal_mask = (_window_label(labels_np, feature_dim) == 0)
        if normal_mask.shape[0] != flat_dataset_tensor.shape[0]:
            minN = min(normal_mask.shape[0], flat_dataset_tensor.shape[0])
            normal_mask = normal_mask[:minN]
            flat_dataset_tensor = flat_dataset_tensor[:minN]
        if normal_mask.any():
            ref_flat = flat_dataset_tensor[normal_mask].detach().cpu().numpy()
        else:
            ref_flat = flat_dataset_tensor.detach().cpu().numpy()

    # 采样以加速
    if ref_flat.shape[0] > max_norm_bank:
        sel = np.random.RandomState(42).choice(ref_flat.shape[0], size=max_norm_bank, replace=False)
        ref_flat = ref_flat[sel]

    normal_ref_WF = ref_flat.reshape(-1, window_size, feature_dim)
    mu_F, sigma_F, ref_corr = fit_normal_reference(normal_ref_WF)

    # 循环两个索引：异常窗、相邻正常窗
    for idx, tag in [(ab_idx, "abnormal"), (norm_idx, "normal")]:
        x_flat = flat_dataset_tensor[idx:idx+1].to(device)         # [1, W*F]
        series_WF = window_to_tensor(x_flat.detach().cpu(), window_size, feature_dim)

        # (a) 注意力 + 原序列 + 标签时间轴
        score, attn = get_attention_on_sample(model, x_flat)
        label_WF = _build_label_WF(labels_np, idx, series_WF)
        plot_attention_and_series(save_dir, f'{tag}_idx{idx}', series_WF, attn_TT=attn, label_WF=label_WF)

        if attn is not None:
            top_idx, top_scores = topk_attended_timesteps(attn, k=min(5, attn.shape[0]))
            np.save(os.path.join(save_dir, f'{tag}_idx{idx}_top_attended.npy'), np.vstack([top_idx, top_scores]))

        # (b) 梯度×输入的重要性 -> 稳健归一后再画图/保存
        sal = grad_input_saliency(model, x_flat)                    # [1, W*F]
        sal_WF = sal.reshape(window_size, feature_dim)
        feat_score_norm, feat_score_raw = _normalize_feat_importance_from_saliency(sal_WF)  # ==== CHANGED ====
        np.save(os.path.join(save_dir, f'{tag}_idx{idx}_saliency.npy'), sal_WF)
        plot_feature_saliency(save_dir, f'{tag}_idx{idx}', feat_score_norm, feature_dim=feature_dim)
        # === NEW: 时间对齐的通道归因（sal_F_weighted） ===
        sal_F_weighted = aggregate_feature_saliency(saliency_TF=sal_WF, attn_TT=attn)
        plt.figure(figsize=(6,4))
        plt.bar(np.arange(feature_dim), sal_F_weighted)
        plt.xlabel('Feature'); plt.ylabel('Importance'); plt.title(f'Feature Saliency (weighted) [{tag}_idx{idx}]')
        plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{tag}_idx{idx}_feat_saliency_weighted.png'), dpi=200); plt.close()

        # === NEW: 时间对齐的统计 Top-K（与 sal_F_weighted 同口径；需要 mu_F/sigma_F/ref_corr 已在函数前面算好） ===
        diff_F_w, z_F_w = weighted_topk_scores(series_WF, mu_F, sigma_F, attn_TT=attn)

        def _bar_topk(values, title, fname, topk=10):
            order = np.argsort(-values)[:min(topk, len(values))]
            plt.figure(figsize=(8,3))
            plt.bar(range(len(order)), values[order])
            plt.xticks(range(len(order)), order)
            plt.xlabel('Feature'); plt.ylabel('Score'); plt.title(title)
            plt.tight_layout(); plt.savefig(os.path.join(save_dir, fname), dpi=200); plt.close()
            return order

        topk_diff_w = _bar_topk(diff_F_w, f'Top-K (weighted |Δ|) [{tag}_idx{idx}]', f'{tag}_idx{idx}_topk_diff_weighted.png')
        topk_z_w    = _bar_topk(z_F_w,    f'Top-K (weighted |z|) [{tag}_idx{idx}]',  f'{tag}_idx{idx}_topk_zscore_weighted.png')

        # === NEW: 打印一致性（Overlap@K + Spearman），方便你在控制台快速核对 ===
        try:
            from scipy.stats import spearmanr
            spearman = lambda a,b: spearmanr(a,b, nan_policy='omit').correlation
        except Exception:
            # 兜底：用秩序列的皮尔逊相关近似 Spearman
            def spearman(a, b):
                ar = np.argsort(np.argsort(-a))
                br = np.argsort(np.argsort(-b))
                ar = (ar - ar.mean())/(ar.std()+1e-12)
                br = (br - br.mean())/(br.std()+1e-12)
                return float(np.dot(ar, br)/len(ar))

        K = min(5, feature_dim)
        sal_top = np.argsort(-sal_F_weighted)[:K]
        ov_diff = len(set(sal_top) & set(topk_diff_w[:K])) / float(K)
        ov_z    = len(set(sal_top) & set(topk_z_w[:K]))    / float(K)
        rho_diff = spearman(sal_F_weighted, diff_F_w)
        rho_z    = spearman(sal_F_weighted, z_F_w)
        print(f'[aligned] {tag}_idx{idx}: overlap@{K} (sal↔|Δ|)={ov_diff:.2f}, (sal↔|z|)={ov_z:.2f};  ρ_s(|Δ|)={rho_diff:.2f},  ρ_s(|z|)={rho_z:.2f}')


    # ---------------- 深度对比（仅做一次，异常窗 vs 相邻正常窗） ----------------
    ab_WF = window_to_tensor(flat_dataset_tensor[ab_idx:ab_idx+1].detach().cpu(), window_size, feature_dim)
    no_WF = window_to_tensor(flat_dataset_tensor[norm_idx:norm_idx+1].detach().cpu(), window_size, feature_dim)

    ab_stats = window_stats_vs_normal(ab_WF, mu_F, sigma_F, ref_corr, normal_bank_WF=normal_ref_WF, save_dir=save_dir)
    no_stats = window_stats_vs_normal(no_WF, mu_F, sigma_F, ref_corr, normal_bank_WF=normal_ref_WF, save_dir=save_dir)

    strong_abnormal = int((ab_stats["z_abs_max_F"] > 3.0).sum() >= 2)

    with open(os.path.join(save_dir, 'summary_deep_compare.txt'), 'w', encoding='utf-8') as f:
        def topk_str(t):
            idx, val = t
            return ", ".join([f"{i}:{v:.2f}" for i, v in zip(idx, val)])
        f.write(
            "=== Deep Compare Summary ===\n"
            f"- Strong-abnormal(by z>3 count>=2): {strong_abnormal}\n"
            f"- Abnormal corr-shift(Fro): {ab_stats['corr_shift_Fro']:.3f}\n"
            f"- Normal   corr-shift(Fro): {no_stats['corr_shift_Fro']:.3f}\n"
            f"- Top-Z features (ab): " + topk_str(ab_stats['top_z_features']) + "\n"
            f"- Top-Δ features (ab): " + topk_str(ab_stats['top_diff_features']) + "\n"
            f"- Top-Z features (no): " + topk_str(no_stats['top_z_features']) + "\n"
            f"- Top-Δ features (no): " + topk_str(no_stats['top_diff_features']) + "\n"
            f"- Nearest-normal distances (if any): "
            + (np.array2string(ab_stats['nearest_normal']['distances'], precision=4)
               if ab_stats['nearest_normal'] else "None") + "\n"
        )

    with open(os.path.join(save_dir, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(
            "生成内容：\n"
            "- *_attn.png：时间×时间的注意力热力图（若模型支持返回注意力）；\n"
            "- *_series.png：对应窗口的多变量曲线；\n"
            "- *_label_timeline.png：若标签能广播/还原才会生成；\n"
            "- *_feat_saliency.png：基于判别器输出的“稳健归一”特征重要性（并保存 *_saliency.npy）；\n"
            "- *_top_attended.npy：Top-K 被关注时间步（索引与分数，若有注意力）；\n"
            "- overlay_ab_vs_nn.png：异常窗与最近正常模板均值叠图；\n"
            "- topk_zscore.png / topk_diff.png：统计显著的前K特征（z-score 与差分突变）；\n"
            "- summary_deep_compare.txt：一页摘要，含强异常判据、相关结构偏移、Top特征等。\n"
        )

# =============================
# 主入口二：True Positive 专用解释（异常窗标红 + 置信度）
# =============================
def _window_label(labels_np, feature_dim):
    """
    接受 [N,F] 或 [N] 的标签，统一成窗口级 [N]（只要任一特征为1就记为该窗异常）
    """
    if labels_np.ndim == 2 and labels_np.shape[1] == feature_dim:
        return (labels_np.sum(axis=1) > 0).astype(int)
    elif labels_np.ndim == 1:
        return labels_np.astype(int)
    else:
        L = labels_np.reshape(labels_np.shape[0], -1).sum(axis=1)
        return (L > 0).astype(int)

@torch.no_grad()
def _score_all(model, flat_dataset_tensor, batch=512):
    """
    用判别器对所有窗口打分（无训练）。返回 numpy [N]
    """
    device = next(model.parameters()).device
    scores = []
    for i in range(0, flat_dataset_tensor.shape[0], batch):
        x = flat_dataset_tensor[i:i+batch].to(device)
        s = model.discriminate(x).detach().cpu().view(-1).numpy()
        scores.append(s)
    scores = np.concatenate(scores, axis=0)
    return scores

def _threshold(scores, method="pot", q=1e-3):
    """
    产生阈值：
      - pot: 调用 pot.py（若失败回退 perc99.5）
      - perc: 直接分位阈值
    返回 (thr, meta)
    """
    try:
        if method == "pot":
            from pot import POT
            pot = POT(scores)
            ret = pot.run()
            thr = ret.get("threshold", np.percentile(scores, 99.5))
            return thr, {"method":"pot", "detail":ret}
        else:
            thr = np.percentile(scores, 99.5)
            return thr, {"method":"percentile", "p":99.5}
    except Exception as e:
        thr = np.percentile(scores, 99.5)
        return thr, {"method":"fallback_perc", "p":99.5, "err":str(e)}

def _plot_pair_windows(save_dir, title_ab, ab_WF, title_no, no_WF, topk_feat=None):
    """
    将异常窗（红色）和正常窗（灰色）分别画图；若给了 topk_feat，则异常窗里那几条用更粗的红线
    """
    os.makedirs(save_dir, exist_ok=True)

    # abnormal（红）
    W, F = ab_WF.shape
    plt.figure(figsize=(10,4))
    for f in range(F):
        lw = 2.2 if (topk_feat is not None and f in set(topk_feat)) else 1.0
        alpha = 0.95 if (topk_feat is not None and f in set(topk_feat)) else 0.8
        plt.plot(ab_WF[:, f], color='r', linewidth=lw, alpha=alpha)
    plt.title(f'Sequence Window (abnormal {title_ab})'); plt.xlabel('Time'); plt.ylabel('Value')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{title_ab}_series_RED.png'), dpi=200); plt.close()

    # normal（灰）
    W2, F2 = no_WF.shape
    plt.figure(figsize=(10,4))
    for f in range(F2):
        plt.plot(no_WF[:, f], color='0.5', alpha=0.8)
    plt.title(f'Sequence Window (normal {title_no})'); plt.xlabel('Time'); plt.ylabel('Value')
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, f'{title_no}_series_GRAY.png'), dpi=200); plt.close()

def _percentile_rank(scores, value):
    return float((scores < value).mean())

def explain_true_positive(
    model,
    flat_dataset_tensor,      # [N, W*F]
    labels_np,                # [N,F] or [N]
    window_size,
    feature_dim,
    out_dir='explain_tp',
    threshold_method="pot",
    topk_feat=5,
    also_draw_attn=True
):
    """
    自动选“模型判异常 + 标签为异常”的TP样本，并选一个TN作对照。
    画红色异常曲线、灰色正常曲线，输出注意力/重要性与置信度摘要。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 长度对齐
    N = min(flat_dataset_tensor.shape[0], labels_np.shape[0])
    flat_dataset_tensor = flat_dataset_tensor[:N]
    labels_np = labels_np[:N]

    # 2) 窗口级标签、分数、阈值与预测
    labels_win = _window_label(labels_np, feature_dim)  # [N]
    scores = _score_all(model, flat_dataset_tensor)     # [N]
    thr, thr_meta = _threshold(scores, method=threshold_method)
    preds = (scores > thr).astype(int)

    # 3) 选择 margin 最大的 TP；选择一个“最靠近阈值”的 TN 作为对照
    tp_candidates = np.where((preds==1) & (labels_win==1))[0]
    if len(tp_candidates) == 0:
        raise RuntimeError("没有找到真阳性(TP)。可放宽阈值或检查标签。")
    margins = scores[tp_candidates] - thr
    tp_idx = int(tp_candidates[np.argmax(margins)])

    tn_candidates = np.where((preds==0) & (labels_win==0))[0]
    tn_idx = int(tn_candidates[np.argmin(scores[tn_candidates]-thr)]) if len(tn_candidates)>0 else None

    # 4) 取窗口数据
    ab_WF = flat_dataset_tensor[tp_idx:tp_idx+1].view(window_size, feature_dim).detach().cpu().numpy()
    if tn_idx is not None:
        no_WF = flat_dataset_tensor[tn_idx:tn_idx+1].view(window_size, feature_dim).detach().cpu().numpy()
    else:
        no_WF = flat_dataset_tensor[0:1].view(window_size, feature_dim).detach().cpu().numpy()

    # 5) saliency & attention  (使用稳健归一后的特征分数)
    sal = grad_input_saliency(model, flat_dataset_tensor[tp_idx:tp_idx+1])
    sal_WF = sal.reshape(window_size, feature_dim)
    feat_score_norm, feat_score_raw = _normalize_feat_importance_from_saliency(sal_WF)  # ==== CHANGED ====
    np.save(os.path.join(out_dir, f'abnormal_idx{tp_idx}_saliency.npy'), sal_WF)

    plt.figure(figsize=(6,4))
    plt.bar(np.arange(feature_dim), feat_score_norm)
    plt.title(f'Feature Saliency'); plt.xlabel('Feature index'); plt.ylabel('Importance')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'abnormal_feat_saliency.png'), dpi=200); plt.close()
    topk_idx = np.argsort(-feat_score_norm)[:min(topk_feat, feature_dim)].tolist()

    if also_draw_attn:
        _, attn = get_attention_on_sample(model, flat_dataset_tensor[tp_idx:tp_idx+1].to(next(model.parameters()).device))
        if attn is not None:
            plt.figure(figsize=(6,5))
            plt.imshow(attn, aspect='auto', origin='lower'); plt.colorbar()
            plt.title(f'Attention Map'); plt.xlabel('Key time'); plt.ylabel('Query time')
            plt.tight_layout(); plt.savefig(os.path.join(out_dir, f'abnormal_attn.png'), dpi=200); plt.close()

    # 6) 正常参考库 & 统计对比
    normal_bank = flat_dataset_tensor[(labels_win==0)]
    if normal_bank.shape[0] == 0:
        normal_bank = flat_dataset_tensor
    # === NEW: TP 的加权通道归因（不等 mu_F/sigma_F） ===
    sal_F_weighted = aggregate_feature_saliency(saliency_TF=sal_WF, attn_TT=attn)
    plt.figure(figsize=(6,4))
    plt.bar(np.arange(feature_dim), sal_F_weighted)
    plt.title('Feature Saliency (weighted)'); plt.xlabel('Feature'); plt.ylabel('Importance')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'abnormal_feat_saliency_weighted.png'), dpi=200); plt.close()
    sal_top_idx_w = np.argsort(-sal_F_weighted)[:min(topk_feat, feature_dim)].tolist()
    normal_ref_WF = normal_bank.detach().cpu().numpy().reshape(-1, window_size, feature_dim)
    mu_F, sigma_F, ref_corr = fit_normal_reference(normal_ref_WF)
    # === NEW: TP 的加权统计（与加权归因口径一致） ===
    diff_F_w, z_F_w = weighted_topk_scores(ab_WF, mu_F, sigma_F, attn_TT=attn)

    def _bar(values, title, fname, topk=10):
        idx = np.argsort(-values)[:min(topk, len(values))]
        plt.figure(figsize=(8,3))
        plt.bar(range(len(idx)), values[idx])
        plt.xticks(range(len(idx)), idx)
        plt.title(title); plt.xlabel('Feature'); plt.ylabel('Score')
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, fname), dpi=200); plt.close()
        return idx

    topk_diff_w = _bar(diff_F_w, 'Top-K (weighted |Δ|)', 'topk_diff_weighted.png')
    topk_z_w    = _bar(z_F_w,    'Top-K (weighted |z|)',  'topk_zscore_weighted.png')

    # 一致性指标（Overlap@K + Spearman），写入 summary
    try:
        from scipy.stats import spearmanr
        spearman = lambda a,b: spearmanr(a,b, nan_policy='omit').correlation
    except Exception:
        def spearman(a, b):
            ar = np.argsort(np.argsort(-a))
            br = np.argsort(np.argsort(-b))
            ar = (ar - ar.mean())/(ar.std()+1e-12)
            br = (br - br.mean())/(br.std()+1e-12)
            return float(np.dot(ar, br)/len(ar))

    K = min(5, feature_dim)
    ov_diff_w = len(set(sal_top_idx_w) & set(topk_diff_w[:K])) / float(K)
    ov_z_w    = len(set(sal_top_idx_w) & set(topk_z_w[:K]))    / float(K)
    rho_diff_w = spearman(sal_F_weighted, diff_F_w)
    rho_z_w    = spearman(sal_F_weighted, z_F_w)

    with open(os.path.join(out_dir, 'tp_explain_summary.txt'), 'a', encoding='utf-8') as f:
        f.write(
            f"--- Weighted (attention-aligned) ---\n"
            f"Top-K salient (weighted): {sal_top_idx_w}\n"
            f"Overlap@{K} (sal↔|Δ| weighted): {ov_diff_w:.2f}\n"
            f"Overlap@{K} (sal↔|z|  weighted): {ov_z_w:.2f}\n"
            f"Spearman(sal, |Δ| weighted): {rho_diff_w:.2f}\n"
            f"Spearman(sal, |z|  weighted): {rho_z_w:.2f}\n"
        )

    ab_stats = window_stats_vs_normal(ab_WF, mu_F, sigma_F, ref_corr, normal_bank_WF=normal_ref_WF, save_dir=out_dir)

    # 7) 画红/灰两窗
    _plot_pair_windows(out_dir, f'abnormal', ab_WF, f'normal', no_WF, topk_feat=topk_idx)

    # 8) 置信度 + 解释一致性摘要
    conf = {
        "score": float(scores[tp_idx]),
        "threshold": float(thr),
        "margin": float(scores[tp_idx] - thr),
        "percentile_rank": float((scores < scores[tp_idx]).mean()),
        "thr_meta": thr_meta
    }
    z_top_idx = np.array(ab_stats["top_z_features"][0], dtype=int)
    diff_top_idx = np.array(ab_stats["top_diff_features"][0], dtype=int)
    sal_top_idx = np.array(topk_idx, dtype=int)
    overlap_z = len(set(z_top_idx).intersection(set(sal_top_idx))) / max(1, len(sal_top_idx))
    overlap_d = len(set(diff_top_idx).intersection(set(sal_top_idx))) / max(1, len(sal_top_idx))

    with open(os.path.join(out_dir, 'tp_explain_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(
            "=== True Positive Explanation ===\n"
            f"TP index: {tp_idx}\n"
            f"TN index: {tn_idx}\n"
            f"Score: {conf['score']:.6f}\n"
            f"Threshold: {conf['threshold']:.6f}\n"
            f"Margin(score-thr): {conf['margin']:.6f}\n"
            f"Global percentile(rank): {conf['percentile_rank']:.3f}\n"
            f"Top-K salient features: {sal_top_idx.tolist()}\n"
            f"Overlap(saliency vs z-top): {overlap_z:.2f}\n"
            f"Overlap(saliency vs diff-top): {overlap_d:.2f}\n"
            f"Corr-shift(Fro): {ab_stats['corr_shift_Fro']:.3f}\n"
        )