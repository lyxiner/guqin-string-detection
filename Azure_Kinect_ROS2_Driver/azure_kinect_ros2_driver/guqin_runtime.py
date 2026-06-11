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


class GuqinRealtimeRuntime:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        sam_guqin_dir: str | None = None,
        threshold: float = 0.5,
        mode: str = "sliding",
        expected_strings: int = 7,
        min_foreground_ratio: float = 0.001,
        recalibration_cooldown_frames: int = 10,
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
        self.checkpoint_path = str(checkpoint)
        self.threshold = float(threshold)
        self.mode = mode
        self.expected_strings = int(expected_strings)
        self.min_foreground_ratio = float(min_foreground_ratio)
        self.recalibration_cooldown_frames = int(recalibration_cooldown_frames)

        self.model = self.load_model(self.checkpoint_path)
        self.predict_fn = (
            self.predict_sliding if self.mode == "sliding" else self.predict_resize
        )
        self.calibrator = self.StringCalibrator(n_strings=self.expected_strings)
        self.tracker: Any | None = None
        self.frame_index = 0
        self.last_recalibration_frame = -10_000

    def _segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prob_map = self.predict_fn(self.model, frame_rgb)
        mask = (prob_map > self.threshold).astype(np.uint8) * 255
        return prob_map.astype(np.float32), mask

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
        self.frame_index += 1
        prob_map, mask = self._segment(frame_bgr)
        mask_ratio = self._mask_ratio(mask)

        if not self._should_try_calibration(mask_ratio):
            return None

        calibrated = False
        recalibrated = False

        if self.tracker is None:
            model = self.calibrator.calibrate(mask)
            self.tracker = self.StringTracker(model)
            self.last_recalibration_frame = self.frame_index
            calibrated = True
        else:
            model = self.tracker.update(mask)
            if getattr(model, "_needs_recalibration", False):
                cooldown_ok = (
                    self.frame_index - self.last_recalibration_frame
                    >= self.recalibration_cooldown_frames
                )
                if cooldown_ok:
                    model = self.calibrator.calibrate(mask)
                    self.tracker = self.StringTracker(model)
                    self.last_recalibration_frame = self.frame_index
                    recalibrated = True

        assert self.tracker is not None
        active_model = self.tracker.model
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
        )

    @staticmethod
    def result_to_json_dict(result: RuntimeFrameResult) -> dict[str, Any]:
        data = asdict(result)
        data.pop("mask", None)
        data.pop("prob_map", None)
        return data
