from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np


def _append_once(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def find_sam_guqin_dir(explicit_dir: str | None = None) -> Path:
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser().resolve())

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "src" / "SAM_guqin")
        candidates.append(parent / "SAM_guqin")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "eval.py").exists() and (candidate / "strings_realtime.py").exists():
            return candidate
    raise FileNotFoundError("could not locate SAM_guqin directory")


@dataclass
class RuntimeFrameResult:
    frame_index: int
    mask: np.ndarray
    prob_map: np.ndarray
    endpoints: list[dict[str, Any]]
    inlier_ratio: float
    n_samples: int
    calibrated: bool
    recalibrated: bool
    needs_recalibration: bool
    mask_foreground_ratio: float
    seg_ms: float = 0.0
    track_ms: float = 0.0
    used_roi: bool = False
    shift_applied_px: float = 0.0


class GuqinRealtimeRuntime:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        sam_guqin_dir: str | None = None,
        threshold: float = 0.5,
        mode: str = "resize",
        expected_strings: int = 7,
        min_foreground_ratio: float = 0.001,
        recalibration_cooldown_frames: int = 10,
        always_recalibrate: bool = False,
        force_recalibrate_every_n: int = 0,
        tracker_max_inlier_dist_px: float = 8.0,
        tracker_recal_inlier_threshold: float = 0.7,
        roi_inference: bool = True,
        roi_inference_pad_px: int = 100,
        lock_theta_on_recalibration: bool = True,
        tracker_smoothing_alpha: float = 0.6,
    ) -> None:
        if mode not in {"resize", "sliding"}:
            raise ValueError(f"unsupported mode: {mode}")

        self.sam_guqin_dir = find_sam_guqin_dir(sam_guqin_dir)
        _append_once(self.sam_guqin_dir)

        eval_module = importlib.import_module("eval")
        realtime_module = importlib.import_module("strings_realtime")
        self.load_model = eval_module.load_model
        self.predict_resize = eval_module.predict_resize
        self.predict_sliding = eval_module.predict_sliding
        self.StringCalibrator = realtime_module.StringCalibrator
        self.StringTracker = realtime_module.StringTracker

        if checkpoint_path is None:
            checkpoint = self.sam_guqin_dir / "checkpoints" / "guqin_best.pth"
        else:
            checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(
                "Guqin model checkpoint was not found: "
                f"{checkpoint}\n"
                "Download or train guqin_best.pth, then place it at "
                "<repo>/SAM_guqin/checkpoints/guqin_best.pth, or run with "
                "--ros-args -p checkpoint_path:=/path/to/guqin_best.pth"
            )
        self.checkpoint_path = str(checkpoint)
        self.threshold = float(threshold)
        self.mode = mode
        self.expected_strings = int(expected_strings)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.recalibration_cooldown_frames = int(recalibration_cooldown_frames)
        self.always_recalibrate = bool(always_recalibrate)
        self.force_recalibrate_every_n = max(0, int(force_recalibrate_every_n))
        self.tracker_max_inlier_dist_px = float(tracker_max_inlier_dist_px)
        self.tracker_recal_inlier_threshold = float(tracker_recal_inlier_threshold)
        self.roi_inference = bool(roi_inference)
        self.roi_inference_pad_px = int(roi_inference_pad_px)
        self.lock_theta_on_recalibration = bool(lock_theta_on_recalibration)
        self.tracker_smoothing_alpha = float(tracker_smoothing_alpha)

        self.model = self.load_model(self.checkpoint_path)
        self.predict_fn = (
            self.predict_sliding if self.mode == "sliding" else self.predict_resize
        )
        self.calibrator = self.StringCalibrator(n_strings=self.expected_strings)
        self.tracker: Any | None = None
        self.frame_index = 0
        self.last_recalibration_frame = -10_000
        self._fg_px_ref: float | None = None  # 全琴可见时的前景像素数基准

    def _new_tracker(self, model: Any) -> Any:
        return self.StringTracker(
            model,
            max_inlier_dist_px=self.tracker_max_inlier_dist_px,
            recal_inlier_threshold=self.tracker_recal_inlier_threshold,
            smoothing_alpha=self.tracker_smoothing_alpha,
        )

    def _roi_from_tracker(self) -> tuple[int, int, int, int] | None:
        """用上一帧 7 根弦的端点算包围盒, 外扩 pad. 返回 (x0, y0, x1, y1)."""
        if self.tracker is None:
            return None
        model = self.tracker.model
        pts = []
        for p0, p1 in model.endpoints():
            pts.append(p0)
            pts.append(p1)
        pts = np.asarray(pts)
        H, W = model.image_hw
        pad = self.roi_inference_pad_px
        x0 = max(0, int(pts[:, 0].min()) - pad)
        x1 = min(W, int(pts[:, 0].max()) + pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        y1 = min(H, int(pts[:, 1].max()) + pad)
        # ROI 太小或退化时放弃, 走全图
        if (x1 - x0) < 64 or (y1 - y0) < 64:
            return None
        return x0, y0, x1, y1

    def _segment(
        self,
        frame_bgr: np.ndarray,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ROI 给定时只对裁剪区推理, 再贴回全尺寸图.

        resize 模式下这一步同时提速和提准: 网络输入分辨率固定,
        喂进去的区域越小, 琴弦占的有效像素越多, mask 越细.
        """
        if roi is not None:
            x0, y0, x1, y1 = roi
            crop_rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            prob_crop = self.predict_fn(self.model, crop_rgb)
            H, W = frame_bgr.shape[:2]
            prob_map = np.zeros((H, W), dtype=np.float32)
            prob_map[y0:y1, x0:x1] = prob_crop.astype(np.float32)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            prob_map = self.predict_fn(self.model, frame_rgb).astype(np.float32)
        mask = (prob_map > self.threshold).astype(np.uint8) * 255
        return prob_map, mask

    def _fast_theta(self, mask: np.ndarray) -> float | None:
        """PCA 估主方向, 几毫秒级, 替代 1.4s 的 Hough.

        弦像素在 mask 里占绝对主导, 前景坐标的主成分方向就是弦方向 d;
        theta 是法线角 (r = x cosθ + y sinθ, 同 skimage Hough 约定),
        由 (-sinθ, cosθ) ∥ d 解出 θ = atan2(-dx, dy), 折回 [-90°, 90°).
        相比锁死旧 theta, 它对琴的旋转是新鲜估计.
        """
        ys, xs = np.nonzero(mask)
        if len(xs) < 200:
            return None
        if len(xs) > 20000:
            idx = np.random.choice(len(xs), 20000, replace=False)
            xs, ys = xs[idx], ys[idx]
        x = xs - xs.mean()
        y = ys - ys.mean()
        cov = np.array([[np.dot(x, x), np.dot(x, y)],
                        [np.dot(x, y), np.dot(y, y)]]) / len(x)
        w, v = np.linalg.eigh(cov)
        dx, dy = v[:, int(np.argmax(w))]  # 主方向 (沿弦)
        theta = float(np.arctan2(-dx, dy))
        if theta >= np.pi / 2:
            theta -= np.pi
        elif theta < -np.pi / 2:
            theta += np.pi
        return theta

    def _align_string_ids(self, new_model: Any, old_model: Any) -> Any:
        """重标定后, 让新模型的编号方向继承旧模型.

        标定层的规范化排序保证了单次结果自洽, 但"弦1在哪一头"
        是个约定; 跨次重标定时必须和旧模型对齐, 否则下游 3D 节点
        和机械臂会拨错弦. 比较新旧模型各自弦1→弦N 的图像坐标走向,
        方向相反就把新模型整组反转重编号.
        """
        def _direction(model: Any) -> float:
            t_ref = 0.5 * (model.t_anchor_lo + model.t_anchor_hi)
            sin_t = np.sin(model.theta_main)
            cos_t = np.cos(model.theta_main)
            uv = []
            for trk in model.tracks:
                r_ref = trk.a + trk.b * t_ref
                uv.append((r_ref * cos_t - t_ref * sin_t,
                           r_ref * sin_t + t_ref * cos_t))
            uv = np.asarray(uv)
            axis = 1 if np.ptp(uv[:, 1]) >= np.ptp(uv[:, 0]) else 0
            return float(uv[-1, axis] - uv[0, axis])

        if _direction(new_model) * _direction(old_model) < 0:
            reversed_tracks = list(reversed(new_model.tracks))
            for i, trk in enumerate(reversed_tracks):
                trk.string_id = i + 1
            new_model.tracks = reversed_tracks
        return new_model

    def _safe_calibrate(self, mask: np.ndarray) -> Any | None:
        """重标定: 用 PCA 快速估 theta 跳过 Hough; 失败退回完整标定."""
        theta_fast = None
        if self.lock_theta_on_recalibration and self.tracker is not None:
            theta_fast = self._fast_theta(mask)
        old_model = self.tracker.model if self.tracker is not None else None
        try:
            model = self.calibrator.calibrate(mask, theta_locked=theta_fast)
        except Exception:
            if theta_fast is not None:
                try:
                    model = self.calibrator.calibrate(mask)
                except Exception:
                    return None
            else:
                return None
        if old_model is not None:
            model = self._align_string_ids(model, old_model)
        return model

    def _mask_ratio(self, mask: np.ndarray) -> float:
        return float((mask > 0).sum() / mask.size)

    def _should_try_calibration(self, mask_ratio: float) -> bool:
        return mask_ratio >= self.min_foreground_ratio

    def _model_to_endpoints(self, model: Any) -> list[dict[str, Any]]:
        payload = []
        for track, (p0, p1) in zip(model.tracks, model.endpoints()):
            payload.append(
                {
                    "string_id": int(track.string_id),
                    "p_start_uv": [float(p0[0]), float(p0[1])],
                    "p_end_uv": [float(p1[0]), float(p1[1])],
                    "track_a": float(track.a),
                    "track_b": float(track.b),
                }
            )
        return payload

    def process_frame(self, frame_bgr: np.ndarray) -> RuntimeFrameResult | None:
        import time as _time

        self.frame_index += 1

        roi = self._roi_from_tracker() if self.roi_inference else None
        t0 = _time.monotonic()
        prob_map, mask = self._segment(frame_bgr, roi=roi)
        seg_ms = (_time.monotonic() - t0) * 1000.0
        mask_ratio = self._mask_ratio(mask)

        # 前景看门狗: ROI 推理下前景像素数明显低于"全琴可见"的基准,
        # 说明琴被移出了旧框 (mask 是半截琴), 立刻退回全图重分割.
        # 拿半截琴去做平移搜索/重拟合正是"挪琴后拟合不上"的根源.
        fg_px = float((mask > 0).sum())
        roi_degraded = (
            roi is not None
            and self._fg_px_ref is not None
            and fg_px < 0.5 * self._fg_px_ref
        )
        if roi is not None and (roi_degraded
                                or not self._should_try_calibration(mask_ratio)):
            roi = None
            t0 = _time.monotonic()
            prob_map, mask = self._segment(frame_bgr, roi=None)
            seg_ms += (_time.monotonic() - t0) * 1000.0
            mask_ratio = self._mask_ratio(mask)
            fg_px = float((mask > 0).sum())

        if not self._should_try_calibration(mask_ratio):
            return None

        calibrated = False
        recalibrated = False
        t0 = _time.monotonic()

        force_periodic_recalibration = (
            self.force_recalibrate_every_n > 0
            and self.frame_index % self.force_recalibrate_every_n == 0
        )

        tracker_was_uninitialized = self.tracker is None
        if tracker_was_uninitialized or self.always_recalibrate or force_periodic_recalibration:
            model = self._safe_calibrate(mask)
            if model is None:
                if tracker_was_uninitialized:
                    return None  # 首帧标定失败, 等下一帧
            else:
                self.tracker = self._new_tracker(model)
                self.last_recalibration_frame = self.frame_index
                if tracker_was_uninitialized:
                    calibrated = True
                else:
                    recalibrated = True
        else:
            model = self.tracker.update(mask)
            # ROI mask 上丢锁: 先全图重分割, 让跟踪器在完整数据上做平移恢复.
            # 这一步成功的话连重标定都不需要.
            if (roi is not None
                    and getattr(model, "_inlier_ratio", 1.0)
                    < self.tracker_recal_inlier_threshold):
                ts = _time.monotonic()
                prob_map, mask = self._segment(frame_bgr, roi=None)
                seg_ms += (_time.monotonic() - ts) * 1000.0
                roi = None
                mask_ratio = self._mask_ratio(mask)
                fg_px = float((mask > 0).sum())
                model = self.tracker.update(mask)
            if getattr(model, "_needs_recalibration", False):
                cooldown_ok = (
                    self.frame_index - self.last_recalibration_frame
                    >= self.recalibration_cooldown_frames
                )
                if cooldown_ok:
                    # 跟踪器丢锁且 ROI 在用: 先全图重分割再标定,
                    # 避免拿半张琴的 mask 去拟合 7 根弦
                    if roi is not None:
                        ts = _time.monotonic()
                        prob_map, mask = self._segment(frame_bgr, roi=None)
                        seg_ms += (_time.monotonic() - ts) * 1000.0
                        roi = None
                    new_model = self._safe_calibrate(mask)
                    if new_model is not None:
                        self.tracker = self._new_tracker(new_model)
                        self.last_recalibration_frame = self.frame_index
                        recalibrated = True

        track_ms = (_time.monotonic() - t0) * 1000.0

        assert self.tracker is not None
        active_model = self.tracker.model

        # 更新"全琴可见"的前景基准: 只在跟踪健康时更新, 避免被异常帧污染
        if (getattr(active_model, "_inlier_ratio", 0.0)
                >= self.tracker_recal_inlier_threshold):
            if self._fg_px_ref is None:
                self._fg_px_ref = fg_px
            else:
                self._fg_px_ref = 0.9 * self._fg_px_ref + 0.1 * fg_px
        return RuntimeFrameResult(
            frame_index=self.frame_index,
            mask=mask,
            prob_map=prob_map,
            endpoints=self._model_to_endpoints(active_model),
            inlier_ratio=float(getattr(active_model, "_inlier_ratio", 1.0)),
            n_samples=int(getattr(active_model, "_n_samples", 0)),
            calibrated=calibrated,
            recalibrated=recalibrated,
            needs_recalibration=bool(
                getattr(active_model, "_needs_recalibration", False)
            ),
            mask_foreground_ratio=mask_ratio,
            seg_ms=seg_ms,
            track_ms=track_ms,
            used_roi=roi is not None,
            shift_applied_px=float(
                getattr(active_model, "_shift_applied_px", 0.0)
            ),
        )

    @staticmethod
    def result_to_json_dict(result: RuntimeFrameResult) -> dict[str, Any]:
        data = asdict(result)
        data.pop("mask", None)
        data.pop("prob_map", None)
        return data