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

    def calibrate(self, mask: np.ndarray,
                  theta_locked: Optional[float] = None) -> StringModel:
        """mask: (H, W) bool/uint8, 返回 StringModel.

        theta_locked: 给定时跳过 estimate_main_theta 的 Hough (~1.4s),
        直接复用旧模型的主方向. 相机不动、琴只平移时 theta 不变,
        重标定耗时从秒级降到几十毫秒.
        """
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

        if theta_locked is None:
            theta = mts.estimate_main_theta(m)
        else:
            theta = float(theta_locked)
        ys, xs, r = mts.project_perpendicular(m, theta)
        t_vals = mts.project_along_string(xs, ys, theta)

        # 透视兜底: 琴窄端远离相机时弦扇形汇聚, 全局 r 直方图
        # 远端的峰互相糊掉, find_seven_centers 找不齐 7 个峰.
        # 此时退到 t 的端部窗口 (弦分得最开的那一段) 重新找峰.
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

        # 规范化弦序: fit_string_tracks 内部每轮迭代都按 r 升序重排,
        # 把 order_top_to_bottom 的结果静默覆盖了. 而 r 与图像 y 的
        # 对应关系取决于 sin(theta) 的符号 —— theta 跨 ±90° 回卷时
        # (琴窄端远离相机带一点旋转就会发生), 弦序会整体翻成 7-1.
        # 这里改按"图像坐标"排序: 在 t 中点处算每根弦的 (u, v),
        # 主跨度在 v 就按 v 升序 (上→下), 在 u 就按 u 升序 (左→右),
        # 与 theta 符号彻底解耦.
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


# ============================== 跟踪层 ==============================

class StringTracker:
    """30Hz 实时跟踪器."""

    def __init__(self, model: StringModel,
                 n_sample_cols: int = 96,
                 roi_pad_px: int = 60,
                 max_inlier_dist_px: float = 8.0,
                 recal_inlier_threshold: float = 0.7,
                 shift_search_px: int = 150,
                 shift_search_step_px: int = 2,
                 smoothing_alpha: float = 0.6):
        """
        n_sample_cols: 沿主轴方向降采样多少列 (96 列 * 7 弦 ≈ 672 个点已经够拟合 7 条直线)
        roi_pad_px: 用上一帧 7 根弦的位置算 ROI, 向两侧外扩这么多像素
        max_inlier_dist_px: 列扫描得到的弦中心与预测 r 的最大允许偏差
        recal_inlier_threshold: 当帧 inlier_ratio 低于此值时, model._needs_recalibration=True,
                               调用方应触发 StringCalibrator 重跑
        shift_search_px: 丢锁时假设琴整体平移, 在 ±shift_search_px 内穷举全局
                         delta_r, 找到 inlier 最多的平移量, 一帧内直接追上
        smoothing_alpha: (a, b) 的 EMA 系数, new = alpha*fit + (1-alpha)*old.
                         1.0 = 关闭平滑; 发生全局平移的那一帧自动跳过平滑
        """
        self.model = model
        self.n_sample_cols = n_sample_cols
        self.roi_pad_px = roi_pad_px
        self.max_inlier_dist_px = max_inlier_dist_px
        self.recal_inlier_threshold = recal_inlier_threshold
        self.shift_search_px = int(shift_search_px)
        self.shift_search_step_px = max(1, int(shift_search_step_px))
        self.smoothing_alpha = float(smoothing_alpha)
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

        琴带旋转时, 远离转轴处的采样点偏差会超过弦间距的一半,
        被指派到相邻弦上, 正负残差在 lstsq 里相互抵消, b 拟不出来 (混叠).
        对策: 第一遍只用 t 中央窗口的样本 (力臂小, 偏差不足以跨弦,
        指派必然正确) 粗估每弦 (a, b); 第二遍用粗估结果重新指派
        全体样本, 再做全跨度精拟合. 每帧迭代一次, 几帧内追上旋转.
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

        琴是刚体: 平移让 7 根弦的 r 同加 delta, 小角度旋转让残差
        随 t 线性变化 (斜率 q). 只搜一维 delta 时, 纯旋转会被迫
        "翻译"成错误的平移 (一侧残差恰好推进邻弦窗口, 评分虚高).
        二维联合搜索后, 旋转归旋转、平移归平移, 互不污染.

        返回 (delta, q, inlier_ratio, coverage). q 以 t_mid 为支点.

        混叠防护: "错位一根弦"的别名也能让 6/7 样本匹配,
        因此评分字典序: 先比有数据支撑的弦数 (coverage), 再比 inlier.
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

        # 早退: 先验证"静止假设" (delta=0, q=0). 绝大多数帧琴没动,
        # inlier 和覆盖都健康就不必跑二维穷举, 静止帧保持毫秒级.
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

        # 预对齐: 运动造成的帧间相干滞后 (平移+小旋转), 先用小范围
        # 二维搜索补掉, 再做指派. 否则模型误差一旦接近半弦距 (~9px),
        # 样本离邻弦反而比离本弦近, 整组被邻弦捕获, 拟合污染成斜线.
        base_tracks = self.model.tracks
        t_mid0 = float(np.median(t_arr)) if len(t_arr) else 0.0
        delta0, q0, ratio0, cov0 = self._estimate_motion(
            t_arr, r_arr, search_px=30, step_px=1)
        # 只有"全部弦都有数据支撑"的运动假设才可信 (允许 1 根被遮挡);
        # 错位 N 根弦的混叠别名必然缺弦, 在这里被挡掉
        if ((abs(delta0) > 0.5 or abs(q0) > 0.005) and ratio0 >= 0.6
                and cov0 >= self.n_strings - 1):
            base_tracks = self._apply_motion(self.model.tracks,
                                             delta0, q0, t_mid0)

        new_tracks, inlier_ratio, t_in = self._assign_and_refit(
            t_arr, r_arr, tracks=base_tracks)
        shift_applied = 0.0

        # 丢锁快速恢复: 琴大概率只是被大幅平移/轻微旋转了.
        # 全图重扫 (旧 ROI 可能已经罩不住新位置), 大范围二维搜索,
        # 若 inlier 明显回升, 直接把 7 根弦一起搬过去, 免去重标定.
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

        # 移动检测: 各弦法向位移的中位数. 在动就跳过平滑 (否则拟合会拖在真弦后面)
        da_med = float(np.median([abs(nt.a - ot.a)
                                  for nt, ot in zip(new_tracks,
                                                    self.model.tracks)]))
        is_moving = (shift_applied != 0.0) or (da_med > 2.0)

        # EMA 平滑: 仅在静止时压逐帧抖动
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

        # t 锚点更新: 琴沿弦方向移动时, 线段端点必须跟着走,
        # 否则 (a, b) 对了, 画出来的线段仍然和真弦错开一截.
        # 扩张立即生效 (出现了更长的弦证据), 收缩用 EMA 慢慢跟 (抗遮挡).
        t_lo, t_hi = self.model.t_anchor_lo, self.model.t_anchor_hi
        if len(t_in) >= 100:
            obs_lo = float(np.quantile(t_in, 0.01))
            obs_hi = float(np.quantile(t_in, 0.99))
            if is_moving:
                t_lo, t_hi = obs_lo, obs_hi
            else:
                t_lo = obs_lo if obs_lo < t_lo else 0.7 * t_lo + 0.3 * obs_lo
                t_hi = obs_hi if obs_hi > t_hi else 0.7 * t_hi + 0.3 * obs_hi

        # 旋转看门狗: 琴整体旋转时, 7 根弦的 b 会一起偏向同一侧.
        # 平移搜索修不了旋转, 必须触发重标定让上层重估 theta.
        b_med = float(np.median([tr.b for tr in new_tracks]))
        theta_drifted = abs(b_med) > 0.02  # ≈ 1.1°

        # 丢锁检测: inlier_ratio 低说明这一帧大部分采样点都不在旧模型预测附近,
        # 平移搜索也救不回来 (可能是旋转/遮挡/分割崩了),
        # 必须由上层触发标定层重跑.
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