"""
古琴琴弦拟合 v5: 智能端点补全 (基于"最长弦"锚点)

核心策略:
  v4 已经能正确拟合每根弦的直线, 端点取自 inlier 沿弦方向 t 的分位.
  现在加上"补全": 找出所有 7 根弦中
    - 沿弦方向 t 最小的那个端点 (t_min_global, 来自最伸向"龙龈"那头的弦)
    - 沿弦方向 t 最大的那个端点 (t_max_global, 来自最伸向"岳山"那头的弦)
  这两个 t 值定义了"理论上的最大弦长". 然后:
    - 每根弦的最终端点 = 该弦直线 在 t = t_min_global 处的点 + t = t_max_global 处的点
    - 这样所有弦的长度都拉到一致, 视觉上都"贯通"

这跟 v2 用边界线相比的优势:
  - v2 假设 7 个端点共线, 在 eval02 上对 (透视下岳山是一条斜线), 在 eval01 上错
    (eval01 中 7 个端点不共线, 因为遮挡严重 + 弦尾自然散开);
  - v5 不假设共线, 只取"最长那根弦能到达的最远 t", 在两种 mask 上都鲁棒.

但这有一个限制需要用户知道:
  - 如果某根弦两端都被遮挡得只剩中间一段, v5 也能正确补全 (因为它沿弦延伸)
  - 如果"最伸向岳山的那根弦"也没真正到岳山 (整体右端都被切了), 那 v5 给的右端就只是
    "可见的最远右端", 不是物理岳山. 这种情况只能靠先验或人工标定解决.
  - 加了一个开关 --no-completion 可以关掉补全, 退回 v4 行为.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.morphology import skeletonize, remove_small_objects
from skimage.transform import hough_line, hough_line_peaks


RANSAC_DIST_PX_DEFAULT = 3.0


# ==================== 圆量平均 (修了 v2 的方向 bug) ====================

def circular_mean_line_angle(angles_rad):
    a = np.asarray(angles_rad, dtype=np.float64)
    s = np.sin(2.0 * a).mean()
    c = np.cos(2.0 * a).mean()
    return float(np.arctan2(s, c) / 2.0)


def estimate_main_theta(mask_bool, n_use=10):
    skel = skeletonize(mask_bool)
    thetas = np.deg2rad(np.linspace(-90, 89.8, 1800))
    h, theta_arr, d_arr = hough_line(skel, theta=thetas)
    accum, angs, _ = hough_line_peaks(h, theta_arr, d_arr,
                                      num_peaks=20,
                                      threshold=0.25 * h.max())
    if len(angs) == 0:
        raise RuntimeError("Hough 找不到直线")
    return circular_mean_line_angle(angs[:min(n_use, len(angs))])


# ==================== r 投影聚类 ====================

def project_perpendicular(mask_bool, theta):
    ys, xs = np.nonzero(mask_bool)
    r = xs * np.cos(theta) + ys * np.sin(theta)
    return ys, xs, r


def find_seven_centers(r_values, n_strings=7):
    r_min, r_max = float(r_values.min()), float(r_values.max())
    n_bins = max(50, int(r_max - r_min) + 1)
    hist, edges = np.histogram(r_values, bins=n_bins, range=(r_min, r_max))
    centers_r = 0.5 * (edges[:-1] + edges[1:])
    k = max(5, n_bins // 80) | 1
    hist_s = np.convolve(hist.astype(float), np.ones(k) / k, mode="same")
    approx_gap = (r_max - r_min) / (n_strings - 1)
    min_dist = max(3, int(approx_gap * 0.5))
    peaks, props = find_peaks(hist_s, distance=min_dist,
                              prominence=hist_s.max() * 0.03)
    if len(peaks) > n_strings:
        order = np.argsort(props["prominences"])[::-1][:n_strings]
        peaks = np.sort(peaks[order])
    elif len(peaks) < n_strings:
        for relax in [0.35, 0.25, 0.15]:
            md = max(2, int(approx_gap * relax))
            peaks2, _ = find_peaks(hist_s, distance=md,
                                   prominence=hist_s.max() * 0.01)
            if len(peaks2) >= n_strings:
                heights = hist_s[peaks2]
                order = np.argsort(heights)[::-1][:n_strings]
                peaks = np.sort(peaks2[order])
                break
    if len(peaks) != n_strings:
        raise RuntimeError(f"只找到 {len(peaks)} 个峰, 期望 {n_strings}")
    return centers_r[peaks]


def order_top_to_bottom(centers_r, theta):
    s = np.sin(theta); c = np.cos(theta)
    if abs(s) < 1e-6:
        return np.argsort(centers_r / c)
    return np.argsort(centers_r / s)


def assign_to_nearest(r_values, centers_r):
    diff = np.abs(r_values[:, None] - centers_r[None, :])
    return np.argmin(diff, axis=1) + 1


# ==================== (t, r) 轨迹拟合 ====================

def project_along_string(xs, ys, theta_main):
    s, c = np.sin(theta_main), np.cos(theta_main)
    return -xs * s + ys * c


def fit_string_track_from_bins(t_values, r_values,
                               bin_size_px=28,
                               min_bin_pts=15):
    if len(t_values) < max(30, 2 * min_bin_pts):
        return None
    t_lo, t_hi = np.quantile(t_values, [0.01, 0.99])
    if (not np.isfinite(t_lo) or not np.isfinite(t_hi) or
            t_hi <= t_lo + 1e-6):
        return None
    if (t_hi - t_lo) < bin_size_px:
        edges = np.linspace(t_lo, t_hi, 3)
    else:
        edges = np.arange(t_lo, t_hi + bin_size_px, bin_size_px)
        if edges[-1] < t_hi:
            edges = np.append(edges, t_hi)

    tt, rr, ww = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t_values >= a) & (t_values < b)
        n = int(m.sum())
        if n < min_bin_pts:
            continue
        tt.append(float(np.median(t_values[m])))
        rr.append(float(np.median(r_values[m])))
        ww.append(float(n))

    if len(tt) < 2:
        return None

    tt = np.asarray(tt, dtype=np.float64)
    rr = np.asarray(rr, dtype=np.float64)
    ww = np.asarray(ww, dtype=np.float64)
    A = np.stack([np.ones_like(tt), tt], axis=1)
    Aw = A * np.sqrt(ww)[:, None]
    bw = rr * np.sqrt(ww)
    coef, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    return {
        "a": float(coef[0]),
        "b": float(coef[1]),
        "n_bins": int(len(tt)),
    }


def assign_to_nearest_tracks(t_values, r_values, intercepts, slopes):
    pred = intercepts[None, :] + t_values[:, None] * slopes[None, :]
    return np.argmin(np.abs(r_values[:, None] - pred), axis=1) + 1


def fit_string_tracks(t_values, r_values, centers_r,
                      n_iter=8, bin_size_px=28, min_bin_pts=15):
    intercepts = np.asarray(centers_r, dtype=np.float64).copy()
    slopes = np.zeros_like(intercepts)
    sid_per_px = assign_to_nearest_tracks(
        t_values, r_values, intercepts, slopes)
    t_ref = float(np.median(t_values))

    for _ in range(n_iter):
        sid_per_px = assign_to_nearest_tracks(
            t_values, r_values, intercepts, slopes)
        tracks = []
        for sid in range(1, len(intercepts) + 1):
            m = (sid_per_px == sid)
            fit = fit_string_track_from_bins(
                t_values[m], r_values[m],
                bin_size_px=bin_size_px,
                min_bin_pts=min_bin_pts)
            if fit is None:
                fit = {
                    "a": float(intercepts[sid - 1]),
                    "b": float(slopes[sid - 1]),
                    "n_bins": 0,
                }
            tracks.append(fit)

        order = np.argsort([
            track["a"] + track["b"] * t_ref for track in tracks
        ])
        intercepts = np.array(
            [tracks[i]["a"] for i in order], dtype=np.float64)
        slopes = np.array(
            [tracks[i]["b"] for i in order], dtype=np.float64)

    sid_per_px = assign_to_nearest_tracks(
        t_values, r_values, intercepts, slopes)
    tracks = []
    for sid in range(1, len(intercepts) + 1):
        m = (sid_per_px == sid)
        fit = fit_string_track_from_bins(
            t_values[m], r_values[m],
            bin_size_px=bin_size_px,
            min_bin_pts=min_bin_pts)
        if fit is None:
            fit = {
                "a": float(intercepts[sid - 1]),
                "b": float(slopes[sid - 1]),
                "n_bins": 0,
            }
        tracks.append(fit)
    return sid_per_px, tracks


def track_ab_to_line(theta_main, a, b):
    nx_m, ny_m = np.cos(theta_main), np.sin(theta_main)
    dx_m, dy_m = -np.sin(theta_main), np.cos(theta_main)
    nx = nx_m - b * dx_m
    ny = ny_m - b * dy_m
    norm = float(np.hypot(nx, ny))
    theta = float(np.arctan2(ny, nx))
    r = float(a / norm)
    return theta, r


def point_on_track_at_t(theta_main, a, b, t):
    r = a + b * t
    u = r * np.cos(theta_main) - t * np.sin(theta_main)
    v = r * np.sin(theta_main) + t * np.cos(theta_main)
    return float(u), float(v)


# ==================== RANSAC 直线拟合 (兜底保留) ====================

def ransac_line_fit(xs, ys, dist_thresh, n_iter=300,
                    min_inlier_ratio=0.15, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(xs)
    if n < 2:
        return None, None, np.zeros(n, dtype=bool)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    best_inliers = None
    best_count = -1
    for _ in range(n_iter):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        norm = np.hypot(d[0], d[1])
        if norm < 1e-6:
            continue
        nx, ny = -d[1] / norm, d[0] / norm
        r0 = nx * p1[0] + ny * p1[1]
        dists = np.abs(pts[:, 0] * nx + pts[:, 1] * ny - r0)
        inliers = dists < dist_thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            if cnt > 0.9 * n:
                break
    if best_inliers is None or best_count < max(2, int(min_inlier_ratio * n)):
        best_inliers = np.ones(n, dtype=bool)
    pin = pts[best_inliers]
    centroid = pin.mean(axis=0)
    cov = np.cov((pin - centroid).T) if len(pin) >= 2 else np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]
    normal = np.array([-direction[1], direction[0]])
    r_fit = float(normal @ centroid)
    theta_fit = float(np.arctan2(normal[1], normal[0]))
    dists = np.abs(pts @ normal - r_fit)
    final_inliers = dists < dist_thresh
    return theta_fit, r_fit, final_inliers


def t_quantile_endpoints(xs, ys, theta_main, q=(0.005, 0.995)):
    """以全局 main_theta 计算每个像素的 t = -x*sin + y*cos,
    返回 inliers 沿弦方向 t 的两个分位 (t_lo, t_hi).
    用 main_theta 而不是每根弦自己的 theta, 保证所有弦的 t 是同一个坐标系."""
    t = project_along_string(xs, ys, theta_main)
    return float(np.quantile(t, q[0])), float(np.quantile(t, q[1]))


def point_on_line_at_t(theta, r, t, theta_main):
    """给定弦直线 (theta, r) 和全局沿弦坐标 t (基于 theta_main),
    返回该 t 在弦上的 (u, v) 像素坐标.

    弦直线: u*cos(theta) + v*sin(theta) = r
    全局沿弦方向: (-sin(theta_main), cos(theta_main))
    取 (u, v) 同时满足上面两个条件即可:
      点 P = O + t * d_main, O 是弦上某点, d_main 是全局沿弦单位向量;
      约束: u*cos(theta_s) + v*sin(theta_s) = r 用来定 O 沿"垂直主方向"的偏移.

    最简单的办法: 弦的法向 (cos(theta), sin(theta)),
                 全局沿弦方向 d_m = (-sin(theta_main), cos(theta_main)).
    设 P = alpha * n_s + t * d_m, 其中 n_s = (cos(theta_s), sin(theta_s)).
    弦方程: P . n_s = r =>  alpha * (n_s.n_s) + t * (d_m.n_s) = r
                      =>  alpha + t * (d_m . n_s) = r   (n_s 是单位向量)
                      =>  alpha = r - t * (d_m . n_s)
    """
    nx_s, ny_s = np.cos(theta), np.sin(theta)
    dx_m, dy_m = -np.sin(theta_main), np.cos(theta_main)
    dm_dot_ns = dx_m * nx_s + dy_m * ny_s
    alpha = r - t * dm_dot_ns
    u = alpha * nx_s + t * dx_m
    v = alpha * ny_s + t * dy_m
    return float(u), float(v)


# ==================== 主流程 ====================

def main(mask_path, orig_path, out_dir,
         n_strings=7, min_size=200,
         ransac_dist_px=RANSAC_DIST_PX_DEFAULT,
         ransac_iter=300,
         do_completion=True):
    os.makedirs(out_dir, exist_ok=True)

    mask = np.array(Image.open(mask_path).convert("L"))
    mask_bool = mask > 127
    mask_bool = remove_small_objects(mask_bool, min_size=min_size)
    H, W = mask_bool.shape
    print(f"[info] mask: {mask_bool.shape}, fg px: {mask_bool.sum()}")

    theta_main = estimate_main_theta(mask_bool)
    print(f"[info] main theta = {np.rad2deg(theta_main):.2f} deg "
          f"(line dir = {np.rad2deg(theta_main)+90:.2f} deg vs horizontal)")

    ys, xs, r_vals = project_perpendicular(mask_bool, theta_main)
    t_vals = project_along_string(xs, ys, theta_main)
    centers_raw = find_seven_centers(r_vals, n_strings=n_strings)
    order = order_top_to_bottom(centers_raw, theta_main)
    centers = centers_raw[order]
    sid_per_px, tracks = fit_string_tracks(
        t_vals, r_vals, centers,
        n_iter=8, bin_size_px=28, min_bin_pts=15)

    # 第一轮: 先在 (t, r) 坐标里拟合每根弦的中心轨迹 r = a + b*t
    rng = np.random.default_rng(0)
    string_data = []
    for sid, track in enumerate(tracks, 1):
        m = (sid_per_px == sid)
        xs_s = xs[m].astype(np.float64)
        ys_s = ys[m].astype(np.float64)
        r_s = r_vals[m].astype(np.float64)
        if len(xs_s) < 20:
            string_data.append(None); continue
        a_f = float(track["a"])
        b_f = float(track["b"])
        th_f, r_f = track_ab_to_line(theta_main, a_f, b_f)
        pred_r = a_f + b_f * t_vals[m]
        inl = np.abs(r_s - pred_r) < ransac_dist_px
        if int(inl.sum()) < 20:
            th_f, r_f, inl = ransac_line_fit(
                xs_s, ys_s, dist_thresh=ransac_dist_px,
                n_iter=ransac_iter, rng=rng)
            a_f = float(r_f)
            b_f = 0.0
        # 注意: 用 main_theta 算 t (统一坐标系), 不用每根弦自己的 theta
        t_lo, t_hi = t_quantile_endpoints(
            xs_s[inl], ys_s[inl], theta_main)
        string_data.append({
            "sid": sid, "theta": th_f, "r": r_f,
            "track_a": a_f, "track_b": b_f,
            "track_n_bins": int(track.get("n_bins", 0)),
            "n_pts": int(len(xs_s)),
            "n_inliers": int(inl.sum()),
            "t_lo_raw": t_lo, "t_hi_raw": t_hi,
        })
        print(f"[info] string{sid}: pts={len(xs_s)} "
              f"inliers={int(inl.sum())} ({inl.mean()*100:.1f}%) "
              f"t_range=[{t_lo:.0f}, {t_hi:.0f}] "
              f"track(a={a_f:.1f}, b={b_f:.5f}, bins={int(track.get('n_bins', 0))})")

    valid = [d for d in string_data if d is not None]

    # 第二轮: 全局补全锚点 = 所有弦中沿弦方向"伸得最远"的两个 t
    if do_completion and len(valid) > 0:
        # 鲁棒地取: t_lo 的 5% 分位 (避免单根弦极端值), t_hi 的 95% 分位
        t_lo_arr = np.array([d["t_lo_raw"] for d in valid])
        t_hi_arr = np.array([d["t_hi_raw"] for d in valid])
        # 真正延伸最远的端点应该是 min(t_lo) 和 max(t_hi),
        # 但用 5%/95% 分位能避免错检测 (比如 mask 把背景误识别成弦)
        t_global_lo = float(np.percentile(t_lo_arr, 5))
        t_global_hi = float(np.percentile(t_hi_arr, 95))
        # 但不要比真实最远更远
        t_global_lo = max(t_global_lo, t_lo_arr.min())
        t_global_hi = min(t_global_hi, t_hi_arr.max())
        print(f"[info] global anchors: t_lo={t_global_lo:.0f}, "
              f"t_hi={t_global_hi:.0f}, "
              f"individual t_lo: {[f'{t:.0f}' for t in t_lo_arr]}, "
              f"individual t_hi: {[f'{t:.0f}' for t in t_hi_arr]}")
    else:
        t_global_lo = t_global_hi = None

    # 算最终端点
    fits = []
    for d in string_data:
        if d is None:
            fits.append(None); continue
        sid, th_s, r_s = d["sid"], d["theta"], d["r"]
        a_s, b_s = d["track_a"], d["track_b"]
        if do_completion:
            t_start = t_global_lo
            t_end = t_global_hi
        else:
            t_start = d["t_lo_raw"]
            t_end = d["t_hi_raw"]
        p_start = point_on_track_at_t(theta_main, a_s, b_s, t_start)
        p_end = point_on_track_at_t(theta_main, a_s, b_s, t_end)
        # raw 端点也算出来供对比
        p_start_raw = point_on_track_at_t(
            theta_main, a_s, b_s, d["t_lo_raw"])
        p_end_raw = point_on_track_at_t(
            theta_main, a_s, b_s, d["t_hi_raw"])
        ext_start = float(np.hypot(p_start[0] - p_start_raw[0],
                                    p_start[1] - p_start_raw[1]))
        ext_end = float(np.hypot(p_end[0] - p_end_raw[0],
                                  p_end[1] - p_end_raw[1]))
        fits.append({
            "string_id": sid,
            "theta_rad": float(th_s),
            "theta_deg": float(np.rad2deg(th_s)),
            "r_pixel": float(r_s),
            "track_a_pixel": float(a_s),
            "track_b_per_t": float(b_s),
            "track_n_bins": int(d["track_n_bins"]),
            "p_start_uv": [p_start[0], p_start[1]],
            "p_end_uv":   [p_end[0],   p_end[1]],
            "p_start_uv_raw": [p_start_raw[0], p_start_raw[1]],
            "p_end_uv_raw":   [p_end_raw[0],   p_end_raw[1]],
            "extension_at_start_px": ext_start,
            "extension_at_end_px":   ext_end,
            "direction_unit": [
                float(-np.sin(th_s)), float(np.cos(th_s))],
            "length_pixel": float(np.hypot(
                p_end[0] - p_start[0], p_end[1] - p_start[1])),
            "n_points": d["n_pts"],
            "n_inliers": d["n_inliers"],
            "ransac_dist_threshold_px": float(ransac_dist_px),
        })
        flag = ""
        if max(ext_start, ext_end) > 60:
            flag = f"   <-- 补全 max ext={max(ext_start, ext_end):.0f}px"
        print(f"[info] string{sid}: len={fits[-1]['length_pixel']:.0f}px"
              f" ext_s={ext_start:.0f} ext_e={ext_end:.0f}{flag}")

    # 叠加到原图
    orig = Image.open(orig_path).convert("RGB")
    if orig.size != (W, H):
        orig = orig.resize((W, H), Image.BILINEAR)
    draw = ImageDraw.Draw(orig)
    palette = [
        (220, 20, 20), (20, 180, 20), (210, 200, 20),
        (40, 40, 220), (180, 30, 200), (20, 200, 200),
        (220, 220, 220),
    ]
    line_w = max(3, int(min(H, W) * 0.0035))
    dot_r = max(8, int(min(H, W) * 0.008))
    for fit, col in zip(fits, palette):
        if fit is None: continue
        u0, v0 = fit["p_start_uv"]; u1, v1 = fit["p_end_uv"]
        draw.line([(u0, v0), (u1, v1)], fill=col, width=line_w)
        for (u, v) in [(u0, v0), (u1, v1)]:
            draw.ellipse([u - dot_r, v - dot_r, u + dot_r, v + dot_r],
                         fill=col, outline=col)
    overlay_path = os.path.join(out_dir, "strings_overlay_on_orig.png")
    orig.save(overlay_path, optimize=True)
    print(f"[ok] overlay -> {overlay_path}")

    # debug
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(mask, cmap="gray")
    for sid, fit in enumerate(fits, 1):
        if fit is None: continue
        col = np.array(palette[sid - 1]) / 255
        u0, v0 = fit["p_start_uv"]; u1, v1 = fit["p_end_uv"]
        ru0, rv0 = fit["p_start_uv_raw"]; ru1, rv1 = fit["p_end_uv_raw"]
        if fit["extension_at_start_px"] > 30:
            ax.plot([u0, ru0], [v0, rv0], "--", color=col, lw=2)
        if fit["extension_at_end_px"] > 30:
            ax.plot([u1, ru1], [v1, rv1], "--", color=col, lw=2)
        ax.plot([ru0, ru1], [rv0, rv1], "-", color=col, lw=2,
                label=f"string{sid}")
        ax.plot([u0, u1], [v0, v1], "o", color=col, ms=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(f"v5 fit + completion (string thresh={ransac_dist_px}px, "
                 f"main_theta={np.rad2deg(theta_main):.1f} deg)")
    ax.legend(loc="upper left", fontsize=8); ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "strings_fit_debug.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)

    out = {
        "image_size_uv": {"width": int(W), "height": int(H)},
        "coordinate_convention": (
            "u = column (x), v = row (y); origin at top-left. "
            "Each string line: u*cos(theta) + v*sin(theta) = r. "
            "Direction unit = (-sin(theta), cos(theta)), p_start -> p_end."
        ),
        "main_theta_rad": float(theta_main),
        "main_theta_deg": float(np.rad2deg(theta_main)),
        "ransac_dist_threshold_px": float(ransac_dist_px),
        "completion_enabled": bool(do_completion),
        "global_t_anchors": {
            "t_lo": float(t_global_lo) if t_global_lo is not None else None,
            "t_hi": float(t_global_hi) if t_global_hi is not None else None,
        },
        "n_strings": n_strings,
        "strings": [f for f in fits if f is not None],
    }
    with open(os.path.join(out_dir, "strings_fit.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[ok] JSON -> {os.path.join(out_dir, 'strings_fit.json')}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default="/home/wz/Git/SAM_guqin/infer_results/eval03_mask.png")
    ap.add_argument("--orig", default="/home/wz/Git/SAM_guqin/eval03.jpg")
    ap.add_argument("--out",  default="/home/wz/Git/SAM_guqin/infer_results")
    ap.add_argument("--ransac-dist-px", type=float,
                    default=RANSAC_DIST_PX_DEFAULT)
    ap.add_argument("--ransac-iter", type=int, default=300)
    ap.add_argument("--min-size", type=int, default=200)
    ap.add_argument("--no-completion", action="store_true",
                    help="Disable end-point completion (debug)")
    args = ap.parse_args()
    main(args.mask, args.orig, args.out,
         ransac_dist_px=args.ransac_dist_px,
         ransac_iter=args.ransac_iter,
         min_size=args.min_size,
         do_completion=not args.no_completion)
