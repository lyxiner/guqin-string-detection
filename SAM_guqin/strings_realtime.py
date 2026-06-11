"""
strings_realtime.py
================================================================
30+ Hz 实时琴弦跟踪.

架构 (标定层 + 跟踪层):
  StringCalibrator   一次性运行: 拿一帧无遮挡 mask, 跑完整 pipeline,
                     得到 main_theta, 7 根弦的初始 (a, b) 轨迹.
                     场景: 启动时, 或每次琴台被推动后用户触发.

  StringTracker      每一帧调用 update(mask):
                     1) ROI 切片 (避开图像大部分黑色区域)
                     2) 列扫描: 按主方向降采样列,
                        每列里把前景像素分成最多 7 组, 取每组中心 y
                     3) 用上一帧的 (a, b) 给每个采样点分配 string_id
                     4) 每根弦的采样点重新 weighted least squares 拟合 (a, b)
                     5) 返回更新后的 7 个 String3D 端点
                     时延目标: < 10 ms / frame

为什么这个架构能稳:
  - main_theta 几乎不变 (相机不动 + 琴只平移). 锁死它就免去了 1.4s 的 Hough;
  - 列扫描比像素枚举快 50 倍, 因为我们只关心"主轴上每个采样位置 7 根弦在哪",
    不需要每个像素都参与;
  - Warm start: 上一帧的 (a, b) 是这一帧的极好初值, 1 轮迭代就收敛;
  - 琴若被推动, (a, b) 跟着平移, 几帧内就追上, 不会丢锁.

接口:
  calibrator.calibrate(mask) -> StringModel
  tracker = StringTracker(model)
  while True:
      mask = get_latest_mask()   # 来自分割模型 / SAM
      strings_2d = tracker.update(mask)   # 7 根弦的 2D 端点
      strings_3d = project_to_3d(strings_2d, K, T_base_cam, z_guqin)
      send_to_moveit(strings_3d)
================================================================
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================== 数据结构 ==============================

@dataclass
class StringTrack:
    """一根弦的 (t, r) 轨迹: r = a + b * t."""
    string_id: int
    a: float
    b: float

    def r_at(self, t: float) -> float:
        return self.a + self.b * t

    def line_uv_at(self, t: float, theta_main: float) -> Tuple[float, float]:
        """给定沿主轴的 t, 返回这根弦在该 t 处的 (u, v) 像素坐标."""
        # 推导见 v5 的 point_on_line_at_t, 这里用 a + b*t 作为 r
        # 弦法向 n_s 跟 main_theta 同向 (一阶近似), 实际有微调
        # 但因为 b 很小 (< 0.05), 误差可以忽略
        nx_m, ny_m = np.cos(theta_main), np.sin(theta_main)
        dx_m, dy_m = -np.sin(theta_main), np.cos(theta_main)
        r = self.a + self.b * t
        # 点 P = r * n_m + t * d_m (主方向参数化)
        u = r * nx_m + t * dx_m
        v = r * ny_m + t * dy_m
        return float(u), float(v)


@dataclass
class StringModel:
    """7 根弦在像素空间的完整模型."""
    theta_main: float                 # 主方向 (法线角, 锁死)
    tracks: List[StringTrack]         # 7 根弦的 (a, b) 轨迹
    t_anchor_lo: float                # 龙龈端 t 锚点
    t_anchor_hi: float                # 岳山端 t 锚点
    image_hw: Tuple[int, int]         # 图像尺寸 (H, W)

    def endpoints(self) -> List[Tuple[Tuple[float, float],
                                       Tuple[float, float]]]:
        """每根弦的 [(p_start_uv, p_end_uv), ...], 顺序 string1..7."""
        out = []
        for tr in self.tracks:
            p0 = tr.line_uv_at(self.t_anchor_lo, self.theta_main)
            p1 = tr.line_uv_at(self.t_anchor_hi, self.theta_main)
            out.append((p0, p1))
        return out


# ============================== 标定层 ==============================

class StringCalibrator:
    """跑一次完整 pipeline 得到 main_theta + 初始 tracks. 慢但只跑一次."""

    def __init__(self, n_strings: int = 7, ransac_dist_px: float = 3.0):
        self.n_strings = n_strings
        self.ransac_dist_px = ransac_dist_px

    def calibrate(self, mask: np.ndarray) -> StringModel:
        """mask: (H, W) bool/uint8, 返回 StringModel."""
        # 复用 mask_to_strings.py 里的完整 pipeline.
        # 这里直接 import 而不是重写, 保证标定结果和离线版本一致.
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__) or ".")
        try:
            import mask_to_strings as mts
        except ImportError:
            # 兼容旧文件名
            import mts

        from skimage.morphology import remove_small_objects
        m = mask > 127 if mask.dtype != bool else mask
        m = remove_small_objects(m.copy(), min_size=200)

        theta = mts.estimate_main_theta(m)
        ys, xs, r = mts.project_perpendicular(m, theta)
        centers_raw = mts.find_seven_centers(r, self.n_strings)
        order = mts.order_top_to_bottom(centers_raw, theta)
        centers = centers_raw[order]
        t_vals = mts.project_along_string(xs, ys, theta)
        sid_per_px, tracks = mts.fit_string_tracks(
            t_vals, r, centers, n_iter=8, bin_size_px=28)

        # 全局 t 锚点 (沿用你版本里的 5%/95% 分位策略)
        t_lo_arr, t_hi_arr = [], []
        for sid in range(1, self.n_strings + 1):
            mm = (sid_per_px == sid)
            if mm.sum() < 30: continue
            tlo, thi = np.quantile(t_vals[mm], [0.005, 0.995])
            t_lo_arr.append(tlo); t_hi_arr.append(thi)
        t_anchor_lo = float(max(np.percentile(t_lo_arr, 5),
                                np.min(t_lo_arr)))
        t_anchor_hi = float(min(np.percentile(t_hi_arr, 95),
                                np.max(t_hi_arr)))

        return StringModel(
            theta_main=float(theta),
            tracks=[StringTrack(string_id=i + 1,
                                 a=float(t["a"]), b=float(t["b"]))
                    for i, t in enumerate(tracks)],
            t_anchor_lo=t_anchor_lo,
            t_anchor_hi=t_anchor_hi,
            image_hw=tuple(m.shape),
        )


# ============================== 跟踪层 ==============================

class StringTracker:
    """30Hz 实时跟踪器."""

    def __init__(self, model: StringModel,
                 n_sample_cols: int = 96,
                 roi_pad_px: int = 60,
                 max_inlier_dist_px: float = 8.0,
                 recal_inlier_threshold: float = 0.7):
        """
        n_sample_cols: 沿主轴方向降采样多少列 (96 列 * 7 弦 ≈ 672 个点已经够拟合 7 条直线)
        roi_pad_px: 用上一帧 7 根弦的位置算 ROI, 向两侧外扩这么多像素
        max_inlier_dist_px: 列扫描得到的弦中心与预测 r 的最大允许偏差
        recal_inlier_threshold: 当帧 inlier_ratio 低于此值时, model._needs_recalibration=True,
                               调用方应触发 StringCalibrator 重跑
        """
        self.model = model
        self.n_sample_cols = n_sample_cols
        self.roi_pad_px = roi_pad_px
        self.max_inlier_dist_px = max_inlier_dist_px
        self.recal_inlier_threshold = recal_inlier_threshold
        # 预计算
        self._theta = model.theta_main
        self._sin = np.sin(self._theta)
        self._cos = np.cos(self._theta)
        self._H, self._W = model.image_hw
        # 主方向是 (-sin, cos), 接近 ±90° 时近似 y 轴; 接近 0° 时近似 x 轴
        # "主轴扫描方向" = 主方向 (沿弦), 我们沿这个方向降采样
        # 对角度判断: cos(theta) 决定弦法向是否更竖, abs(cos) 大 -> 弦近垂直
        # 对实际两张图 (theta=33° 或 89°), 弦都是近水平的, 所以沿 x 扫
        # 简单粗暴: 哪个轴方向跨度大就沿哪个轴扫, 用 |dx_m| > |dy_m| 判断
        self._scan_along_x = abs(-self._sin) > abs(self._cos)

    # ---------- 列/行扫描 (核心快路径) ----------

    def _scan_columns(self, mask: np.ndarray,
                       y_lo: int, y_hi: int) -> Tuple[np.ndarray, np.ndarray]:
        """沿 x 方向降采样列, 每列把 mask 前景像素分组成最多 7 组,
        每组取中心 y. 返回 (col_x_array, list_of_y_arrays).

        但为了向量化, 我们 flatten 成 (sample_t, sample_r) 两个数组.
        """
        H_roi = y_hi - y_lo
        sample_xs = np.linspace(0, self._W - 1,
                                self.n_sample_cols).astype(np.int32)
        all_t, all_r = [], []
        for x in sample_xs:
            col = mask[y_lo:y_hi, x]
            ys_local = np.where(col)[0]
            if len(ys_local) == 0:
                continue
            # 把连续的 y 分组 (gap > 3 切开)
            ys_full = ys_local + y_lo
            breaks = np.where(np.diff(ys_full) > 3)[0] + 1
            groups = np.split(ys_full, breaks)
            for g in groups:
                cy = (g[0] + g[-1]) * 0.5
                # 算 (t, r): t = -x*sin + y*cos, r = x*cos + y*sin
                t = -x * self._sin + cy * self._cos
                r = x * self._cos + cy * self._sin
                all_t.append(t); all_r.append(r)
        if not all_t:
            return np.empty(0), np.empty(0)
        return np.asarray(all_t), np.asarray(all_r)

    def _scan_rows(self, mask: np.ndarray,
                    x_lo: int, x_hi: int) -> Tuple[np.ndarray, np.ndarray]:
        """弦近垂直时用行扫描."""
        sample_ys = np.linspace(0, self._H - 1,
                                self.n_sample_cols).astype(np.int32)
        all_t, all_r = [], []
        for y in sample_ys:
            row = mask[y, x_lo:x_hi]
            xs_local = np.where(row)[0]
            if len(xs_local) == 0:
                continue
            xs_full = xs_local + x_lo
            breaks = np.where(np.diff(xs_full) > 3)[0] + 1
            groups = np.split(xs_full, breaks)
            for g in groups:
                cx = (g[0] + g[-1]) * 0.5
                t = -cx * self._sin + y * self._cos
                r = cx * self._cos + y * self._sin
                all_t.append(t); all_r.append(r)
        if not all_t:
            return np.empty(0), np.empty(0)
        return np.asarray(all_t), np.asarray(all_r)

    # ---------- ROI 估计 ----------

    def _compute_roi(self) -> Tuple[int, int]:
        """根据当前 7 根弦的 (a, b), 算图像里 ROI 的范围."""
        # 在主轴方向上, ROI 横跨 t_anchor_lo .. t_anchor_hi
        # 在法线方向上, ROI 是 7 根弦 r 的范围 + roi_pad_px
        ts = np.linspace(self.model.t_anchor_lo,
                         self.model.t_anchor_hi, 8)
        all_uv = []
        for tr in self.model.tracks:
            for t in ts:
                u, v = tr.line_uv_at(t, self._theta)
                all_uv.append((u, v))
        all_uv = np.asarray(all_uv)
        if self._scan_along_x:
            v_min, v_max = all_uv[:, 1].min(), all_uv[:, 1].max()
            return (max(0, int(v_min) - self.roi_pad_px),
                    min(self._H, int(v_max) + self.roi_pad_px))
        else:
            u_min, u_max = all_uv[:, 0].min(), all_uv[:, 0].max()
            return (max(0, int(u_min) - self.roi_pad_px),
                    min(self._W, int(u_max) + self.roi_pad_px))

    # ---------- 弦更新 ----------

    def _assign_and_refit(self, t_arr: np.ndarray,
                           r_arr: np.ndarray
                           ) -> Tuple[List[StringTrack], float]:
        """每个采样点 (t, r) 分给最近的弦, 然后每根弦重拟合 (a, b).
        返回 (new_tracks, inlier_ratio).

        inlier_ratio 低 (< 0.3) 表示"采样点跟旧模型不匹配", 通常意味着琴大幅移动.
        调用方应据此触发重新标定.
        """
        if len(t_arr) == 0:
            return self.model.tracks, 0.0

        a_arr = np.array([tr.a for tr in self.model.tracks])
        b_arr = np.array([tr.b for tr in self.model.tracks])
        pred = a_arr[None, :] + t_arr[:, None] * b_arr[None, :]
        dr = np.abs(r_arr[:, None] - pred)
        sid = np.argmin(dr, axis=1)
        min_dr = dr[np.arange(len(t_arr)), sid]
        valid = min_dr < self.max_inlier_dist_px
        inlier_ratio = float(valid.mean())

        new_tracks = []
        for k in range(self.n_strings):
            mm = valid & (sid == k)
            if mm.sum() < 4:
                new_tracks.append(self.model.tracks[k])
                continue
            tt = t_arr[mm]; rr = r_arr[mm]
            A = np.stack([np.ones_like(tt), tt], axis=1)
            coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
            new_tracks.append(StringTrack(string_id=k + 1,
                                           a=float(coef[0]),
                                           b=float(coef[1])))
        return new_tracks, inlier_ratio

    @property
    def n_strings(self) -> int:
        return len(self.model.tracks)

    # ---------- 主入口 ----------

    def update(self, mask: np.ndarray) -> StringModel:
        """处理一帧 mask, 返回更新后的 StringModel.

        如果模型 ._needs_recalibration = True, 表明这一帧丢锁严重,
        调用方应该用这一帧 mask 重新跑 StringCalibrator.calibrate()
        并构造新的 StringTracker.
        """
        if mask.dtype != bool:
            mask = mask > 127

        if self._scan_along_x:
            y_lo, y_hi = self._compute_roi()
            t_arr, r_arr = self._scan_columns(mask, y_lo, y_hi)
        else:
            x_lo, x_hi = self._compute_roi()
            t_arr, r_arr = self._scan_rows(mask, x_lo, x_hi)

        new_tracks, inlier_ratio = self._assign_and_refit(t_arr, r_arr)

        # 丢锁检测: inlier_ratio 低说明这一帧大部分采样点都不在旧模型预测附近,
        # 通常是琴台被大幅推动了. 这种情况跟踪层无法仅用增量更新追上,
        # 必须由上层触发标定层重跑.
        needs_recal = (inlier_ratio < self.recal_inlier_threshold and
                       len(t_arr) > 50)

        self.model = StringModel(
            theta_main=self.model.theta_main,
            tracks=new_tracks,
            t_anchor_lo=self.model.t_anchor_lo,
            t_anchor_hi=self.model.t_anchor_hi,
            image_hw=self.model.image_hw,
        )
        self.model._inlier_ratio = inlier_ratio
        self.model._needs_recalibration = needs_recal
        self.model._n_samples = int(len(t_arr))
        return self.model


# ============================== 测试 / benchmark ==============================

if __name__ == "__main__":
    import sys
    from PIL import Image
    sys.path.insert(0, "/home/claude")

    print("=== Calibration phase (run once) ===")
    mask = np.array(Image.open("/home/claude/mask01.png").convert("L"))
    cal = StringCalibrator()
    t0 = time.time()
    model = cal.calibrate(mask)
    t_cal = (time.time() - t0) * 1000
    print(f"Calibration time: {t_cal:.0f} ms")
    print(f"  theta_main = {np.rad2deg(model.theta_main):.2f} deg")
    print(f"  t_anchors  = ({model.t_anchor_lo:.0f}, {model.t_anchor_hi:.0f})")
    for tr in model.tracks:
        print(f"  string{tr.string_id}: a={tr.a:.2f} b={tr.b:.5f}")

    print("\n=== Tracking phase (per-frame) ===")
    tracker = StringTracker(model)
    # 跑 50 帧 (用同一张 mask 模拟; 真实场景每帧来自相机分割)
    times = []
    for i in range(50):
        t0 = time.time()
        new_model = tracker.update(mask)
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f"Per-frame time over 50 frames:")
    print(f"  mean   = {times.mean():.2f} ms")
    print(f"  median = {np.median(times):.2f} ms")
    print(f"  p95    = {np.percentile(times, 95):.2f} ms")
    print(f"  p99    = {np.percentile(times, 99):.2f} ms")
    print(f"  -> {1000.0 / times.mean():.0f} Hz")

    # 验证跟踪结果跟标定层一致 (没漂移)
    print("\n=== Sanity check: tracker output matches calibrator ===")
    for old_tr, new_tr in zip(model.tracks, new_model.tracks):
        print(f"  string{old_tr.string_id}: "
              f"a {old_tr.a:.2f} -> {new_tr.a:.2f}  "
              f"(da={new_tr.a-old_tr.a:+.3f}), "
              f"b {old_tr.b:.5f} -> {new_tr.b:.5f}")
