import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
        nx_m, ny_m = np.cos(theta_main), np.sin(theta_main)
        dx_m, dy_m = -np.sin(theta_main), np.cos(theta_main)
        r = self.a + self.b * t
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


class StringCalibrator:
    """从二值 mask 中标定七根弦的初始模型."""

    def __init__(self, n_strings: int = 7, ransac_dist_px: float = 3.0):
        self.n_strings = n_strings
        self.ransac_dist_px = ransac_dist_px

    def calibrate(self, mask: np.ndarray,
                  theta_locked: Optional[float] = None) -> StringModel:
        """mask: (H, W) bool/uint8, 返回 StringModel.

        theta_locked: 给定时复用旧模型的主方向, 加快重标定.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__) or ".")
        try:
            import mask_to_strings as mts
        except ImportError:
            import mts

        from skimage.morphology import remove_small_objects
        m = mask > 127 if mask.dtype != bool else mask
        m = remove_small_objects(m.copy(), min_size=200)

        if theta_locked is None:
            theta = mts.estimate_main_theta(m)
        else:
            theta = float(theta_locked)
        ys, xs, r = mts.project_perpendicular(m, theta)
        t_vals = mts.project_along_string(xs, ys, theta)

        # 透视明显时, 端部窗口比全局 r 直方图更容易分出 7 个峰.
        centers_raw = None
        try:
            centers_raw = mts.find_seven_centers(r, self.n_strings)
        except RuntimeError:
            t_q30, t_q70 = np.quantile(t_vals, [0.3, 0.7])
            for band in ((t_vals <= t_q30), (t_vals >= t_q70)):
                if band.sum() < 300:
                    continue
                try:
                    centers_raw = mts.find_seven_centers(
                        r[band], self.n_strings)
                    break
                except RuntimeError:
                    continue
        if centers_raw is None:
            raise RuntimeError(
                f"find_seven_centers failed on full mask and both t-bands")

        order = mts.order_top_to_bottom(centers_raw, theta)
        centers = centers_raw[order]
        sid_per_px, tracks = mts.fit_string_tracks(
            t_vals, r, centers, n_iter=8, bin_size_px=28)

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

        # 按图像坐标重新排序, 避免 theta 回卷导致弦序翻转.
        t_ref = 0.5 * (t_anchor_lo + t_anchor_hi)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        uv = []
        for trk in tracks:
            r_ref = float(trk["a"]) + float(trk["b"]) * t_ref
            uv.append((r_ref * cos_t - t_ref * sin_t,
                       r_ref * sin_t + t_ref * cos_t))
        uv = np.asarray(uv)
        axis = 1 if np.ptp(uv[:, 1]) >= np.ptp(uv[:, 0]) else 0
        canon = np.argsort(uv[:, axis])
        tracks = [tracks[i] for i in canon]

        return StringModel(
            theta_main=float(theta),
            tracks=[StringTrack(string_id=i + 1,
                                 a=float(t["a"]), b=float(t["b"]))
                    for i, t in enumerate(tracks)],
            t_anchor_lo=t_anchor_lo,
            t_anchor_hi=t_anchor_hi,
            image_hw=tuple(m.shape),
        )


class StringTracker:
    """实时更新七根弦的像素模型."""

    def __init__(self, model: StringModel,
                 n_sample_cols: int = 96,
                 roi_pad_px: int = 60,
                 max_inlier_dist_px: float = 8.0,
                 recal_inlier_threshold: float = 0.7,
                 shift_search_px: int = 150,
                 shift_search_step_px: int = 2,
                 smoothing_alpha: float = 0.6):
        self.model = model
        self.n_sample_cols = n_sample_cols
        self.roi_pad_px = roi_pad_px
        self.max_inlier_dist_px = max_inlier_dist_px
        self.recal_inlier_threshold = recal_inlier_threshold
        self.shift_search_px = int(shift_search_px)
        self.shift_search_step_px = max(1, int(shift_search_step_px))
        self.smoothing_alpha = float(smoothing_alpha)
        self._theta = model.theta_main
        self._sin = np.sin(self._theta)
        self._cos = np.cos(self._theta)
        self._H, self._W = model.image_hw
        self._scan_along_x = abs(-self._sin) > abs(self._cos)

    def _scan_columns(self, mask: np.ndarray,
                       y_lo: int, y_hi: int) -> Tuple[np.ndarray, np.ndarray]:
        """沿 x 方向降采样, 返回采样中心点的 (t, r)."""
        H_roi = y_hi - y_lo
        sample_xs = np.linspace(0, self._W - 1,
                                self.n_sample_cols).astype(np.int32)
        all_t, all_r = [], []
        for x in sample_xs:
            col = mask[y_lo:y_hi, x]
            ys_local = np.where(col)[0]
            if len(ys_local) == 0:
                continue
            ys_full = ys_local + y_lo
            breaks = np.where(np.diff(ys_full) > 3)[0] + 1
            groups = np.split(ys_full, breaks)
            for g in groups:
                cy = (g[0] + g[-1]) * 0.5
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

    def _compute_roi(self) -> Tuple[int, int]:
        """根据当前 7 根弦的 (a, b), 算图像里 ROI 的范围."""
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

    def _fit_once(self, t_arr: np.ndarray, r_arr: np.ndarray,
                  tracks: List[StringTrack]
                  ) -> Tuple[List[StringTrack], float, np.ndarray]:
        """单遍: 指派 -> 每弦 lstsq. 返回 (tracks, inlier_ratio, inlier_t)."""
        a_arr = np.array([tr.a for tr in tracks])
        b_arr = np.array([tr.b for tr in tracks])
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
                new_tracks.append(tracks[k])
                continue
            tt = t_arr[mm]; rr = r_arr[mm]
            A = np.stack([np.ones_like(tt), tt], axis=1)
            coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
            new_tracks.append(StringTrack(string_id=k + 1,
                                           a=float(coef[0]),
                                           b=float(coef[1])))
        return new_tracks, inlier_ratio, t_arr[valid]

    def _assign_and_refit(self, t_arr: np.ndarray,
                           r_arr: np.ndarray,
                           tracks: Optional[List[StringTrack]] = None
                           ) -> Tuple[List[StringTrack], float, np.ndarray]:
        """两遍拟合.

        先用 t 中央窗口粗估, 再用全体样本精拟合, 降低旋转场景下的错配。
        """
        if tracks is None:
            tracks = self.model.tracks
        if len(t_arr) == 0:
            return tracks, 0.0, np.empty(0)

        if len(t_arr) >= 50:
            t_mid = float(np.median(t_arr))
            span = float(np.quantile(t_arr, 0.9) - np.quantile(t_arr, 0.1))
            central = np.abs(t_arr - t_mid) < 0.25 * max(span, 1.0)
            if central.sum() >= 30:
                tracks, _, _ = self._fit_once(
                    t_arr[central], r_arr[central], tracks)

        return self._fit_once(t_arr, r_arr, tracks)

    def _estimate_motion(self, t_arr: np.ndarray,
                          r_arr: np.ndarray,
                          search_px: Optional[int] = None,
                          step_px: Optional[int] = None,
                          q_max: float = 0.05,
                          q_step: float = 0.01
                          ) -> Tuple[float, float, float, int]:
        """刚体运动假设下的 (平移 delta, 旋转斜率 q) 二维穷举.

        返回 (delta, q, inlier_ratio, coverage). q 以 t_mid 为支点.
        """
        if len(t_arr) < 30:
            return 0.0, 0.0, 0.0, 0
        search = self.shift_search_px if search_px is None else int(search_px)
        step = self.shift_search_step_px if step_px is None else max(1, int(step_px))
        a_arr = np.array([tr.a for tr in self.model.tracks])
        b_arr = np.array([tr.b for tr in self.model.tracks])
        pred = a_arr[None, :] + t_arr[:, None] * b_arr[None, :]
        dr0 = r_arr[:, None] - pred                           # (N, 7)
        t_mid = float(np.median(t_arr))
        tc = (t_arr - t_mid)[:, None]                         # (N, 1)

        # 静止帧直接返回, 避免不必要的二维搜索。
        hit0 = np.abs(dr0) < self.max_inlier_dist_px
        ratio_0 = float(hit0.any(axis=1).mean())
        cov_0 = int((hit0.sum(axis=0) >= 8).sum())
        if ratio_0 >= 0.9 and cov_0 >= self.n_strings - 1:
            return 0.0, 0.0, ratio_0, cov_0

        deltas = np.arange(-search, search + 1, step, dtype=np.float64)
        qs = np.arange(-q_max, q_max + q_step / 2, q_step)

        best = (-1.0, 0.0, 0.0, 0.0, 0)  # score, delta, q, ratio, cov
        for q in qs:
            res = dr0 - q * tc                                # (N, 7)
            dist = np.abs(res[None, :, :] - deltas[:, None, None])
            hit = dist < self.max_inlier_dist_px              # (D, N, 7)
            inl = hit.any(axis=2).mean(axis=1)                # (D,)
            coverage = (hit.sum(axis=1) >= 8).sum(axis=1)     # (D,)
            score = coverage * 10.0 + inl
            k = int(np.argmax(score))
            if score[k] > best[0]:
                best = (float(score[k]), float(deltas[k]), float(q),
                        float(inl[k]), int(coverage[k]))
        _, delta, q, ratio, cov = best
        return delta, q, ratio, cov

    def _apply_motion(self, tracks: List[StringTrack],
                       delta: float, q: float,
                       t_mid: float) -> List[StringTrack]:
        """把 (delta, q) 折进 (a, b): r = a + bt + delta + q(t - t_mid)."""
        return [StringTrack(tr.string_id,
                            a=tr.a + delta - q * t_mid,
                            b=tr.b + q)
                for tr in tracks]

    @property
    def n_strings(self) -> int:
        return len(self.model.tracks)

    def update(self, mask: np.ndarray) -> StringModel:
        """处理一帧 mask, 返回更新后的 StringModel.

        丢锁严重时会在返回模型上设置 _needs_recalibration。
        """
        if mask.dtype != bool:
            mask = mask > 127

        if self._scan_along_x:
            y_lo, y_hi = self._compute_roi()
            t_arr, r_arr = self._scan_columns(mask, y_lo, y_hi)
        else:
            x_lo, x_hi = self._compute_roi()
            t_arr, r_arr = self._scan_rows(mask, x_lo, x_hi)

        base_tracks = self.model.tracks
        t_mid0 = float(np.median(t_arr)) if len(t_arr) else 0.0
        delta0, q0, ratio0, cov0 = self._estimate_motion(
            t_arr, r_arr, search_px=30, step_px=1)
        if ((abs(delta0) > 0.5 or abs(q0) > 0.005) and ratio0 >= 0.6
                and cov0 >= self.n_strings - 1):
            base_tracks = self._apply_motion(self.model.tracks,
                                             delta0, q0, t_mid0)

        new_tracks, inlier_ratio, t_in = self._assign_and_refit(
            t_arr, r_arr, tracks=base_tracks)
        shift_applied = 0.0

        # ROI 丢锁时用全图重扫尝试恢复大幅平移。
        if inlier_ratio < self.recal_inlier_threshold:
            if self._scan_along_x:
                t_full, r_full = self._scan_columns(mask, 0, self._H)
            else:
                t_full, r_full = self._scan_rows(mask, 0, self._W)
            delta, q, ratio_at_delta, cov = self._estimate_motion(
                t_full, r_full)
            if (ratio_at_delta > max(inlier_ratio + 0.15, 0.5)
                    and cov >= self.n_strings - 1
                    and (abs(delta) > 1.0 or abs(q) > 0.005)):
                t_mid_f = float(np.median(t_full))
                shifted = self._apply_motion(self.model.tracks,
                                             delta, q, t_mid_f)
                new_tracks, inlier_ratio, t_in = self._assign_and_refit(
                    t_full, r_full, tracks=shifted)
                t_arr = t_full
                shift_applied = delta

        da_med = float(np.median([abs(nt.a - ot.a)
                                  for nt, ot in zip(new_tracks,
                                                    self.model.tracks)]))
        is_moving = (shift_applied != 0.0) or (da_med > 2.0)

        if self.smoothing_alpha < 1.0 and not is_moving:
            alpha = self.smoothing_alpha
            new_tracks = [
                StringTrack(
                    string_id=nt.string_id,
                    a=alpha * nt.a + (1.0 - alpha) * ot.a,
                    b=alpha * nt.b + (1.0 - alpha) * ot.b,
                )
                for nt, ot in zip(new_tracks, self.model.tracks)
            ]

        t_lo, t_hi = self.model.t_anchor_lo, self.model.t_anchor_hi
        if len(t_in) >= 100:
            obs_lo = float(np.quantile(t_in, 0.01))
            obs_hi = float(np.quantile(t_in, 0.99))
            if is_moving:
                t_lo, t_hi = obs_lo, obs_hi
            else:
                t_lo = obs_lo if obs_lo < t_lo else 0.7 * t_lo + 0.3 * obs_lo
                t_hi = obs_hi if obs_hi > t_hi else 0.7 * t_hi + 0.3 * obs_hi

        b_med = float(np.median([tr.b for tr in new_tracks]))
        theta_drifted = abs(b_med) > 0.02

        needs_recal = ((inlier_ratio < self.recal_inlier_threshold and
                        len(t_arr) > 50)
                       or (theta_drifted and len(t_arr) > 50))

        self.model = StringModel(
            theta_main=self.model.theta_main,
            tracks=new_tracks,
            t_anchor_lo=t_lo,
            t_anchor_hi=t_hi,
            image_hw=self.model.image_hw,
        )
        self.model._inlier_ratio = inlier_ratio
        self.model._needs_recalibration = needs_recal
        self.model._n_samples = int(len(t_arr))
        self.model._shift_applied_px = float(shift_applied)
        self.model._theta_drifted = theta_drifted
        return self.model

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Binary string mask image")
    args = parser.parse_args()

    print("=== Calibration phase (run once) ===")
    mask = np.array(Image.open(args.mask).convert("L"))
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

    print("\n=== Sanity check: tracker output matches calibrator ===")
    for old_tr, new_tr in zip(model.tracks, new_model.tracks):
        print(f"  string{old_tr.string_id}: "
              f"a {old_tr.a:.2f} -> {new_tr.a:.2f}  "
              f"(da={new_tr.a-old_tr.a:+.3f}), "
              f"b {old_tr.b:.5f} -> {new_tr.b:.5f}")
