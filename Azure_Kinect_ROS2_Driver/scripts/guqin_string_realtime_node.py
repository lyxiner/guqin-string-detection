#!/usr/bin/env python3

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from azure_kinect_ros2_driver.guqin_runtime import (
    GuqinRealtimeRuntime,
    find_sam_guqin_dir,
)


class GuqinStringRealtimeNode(Node):
    def __init__(self) -> None:
        super().__init__("guqin_string_realtime_node")
        self.bridge = CvBridge()

        self.declare_parameter(
            "image_topic",
            "/k4a/rgb/image_raw",
            ParameterDescriptor(description="Input RGB image topic"),
        )
        self.declare_parameter(
            "sam_guqin_dir",
            str(find_sam_guqin_dir()),
            ParameterDescriptor(description="SAM_guqin source directory"),
        )
        self.declare_parameter(
            "checkpoint_path",
            str((find_sam_guqin_dir() / "checkpoints" / "guqin_best.pth").resolve()),
            ParameterDescriptor(description="UNet checkpoint path"),
        )
        self.declare_parameter(
            "inference_mode",
            "resize",
            ParameterDescriptor(description="resize or sliding (resize is ~10x faster)"),
        )
        self.declare_parameter(
            "mask_threshold",
            0.5,
            ParameterDescriptor(description="Segmentation threshold"),
        )
        self.declare_parameter(
            "expected_strings",
            7,
            ParameterDescriptor(description="Expected string count"),
        )
        self.declare_parameter(
            "always_recalibrate",
            False,
            ParameterDescriptor(description="Run full string calibration on every valid frame"),
        )
        self.declare_parameter(
            "force_recalibrate_every_n",
            0,
            ParameterDescriptor(description="Run full calibration every N valid frames, 0 disables"),
        )
        self.declare_parameter(
            "tracker_max_inlier_dist_px",
            8.0,
            ParameterDescriptor(description="Maximum tracker assignment distance in pixels"),
        )
        self.declare_parameter(
            "tracker_recal_inlier_threshold",
            0.7,
            ParameterDescriptor(description="Recalibrate when tracker inlier ratio is below this"),
        )
        self.declare_parameter(
            "publish_mask_topic",
            "/guqin/strings_mask",
            ParameterDescriptor(description="Output mask topic"),
        )
        self.declare_parameter(
            "publish_json_topic",
            "/guqin/strings_fit_json",
            ParameterDescriptor(description="Output JSON topic"),
        )
        self.declare_parameter(
            "publish_overlay_topic",
            "/guqin/strings_overlay",
            ParameterDescriptor(description="Output overlay image topic"),
        )
        self.declare_parameter(
            "debug_overlay_dir",
            "",
            ParameterDescriptor(description="Optional directory for debug overlays"),
        )
        self.declare_parameter(
            "save_debug_every_n",
            0,
            ParameterDescriptor(description="Save debug overlay every N frames, 0 disables"),
        )

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        sam_guqin_dir = self.get_parameter("sam_guqin_dir").get_parameter_value().string_value
        checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value
        inference_mode = self.get_parameter("inference_mode").get_parameter_value().string_value
        mask_threshold = self.get_parameter("mask_threshold").get_parameter_value().double_value
        expected_strings = self.get_parameter("expected_strings").get_parameter_value().integer_value
        always_recalibrate = self.get_parameter("always_recalibrate").get_parameter_value().bool_value
        force_recalibrate_every_n = self.get_parameter("force_recalibrate_every_n").get_parameter_value().integer_value
        tracker_max_inlier_dist_px = self.get_parameter("tracker_max_inlier_dist_px").get_parameter_value().double_value
        tracker_recal_inlier_threshold = self.get_parameter("tracker_recal_inlier_threshold").get_parameter_value().double_value
        publish_mask_topic = self.get_parameter("publish_mask_topic").get_parameter_value().string_value
        publish_json_topic = self.get_parameter("publish_json_topic").get_parameter_value().string_value
        publish_overlay_topic = self.get_parameter("publish_overlay_topic").get_parameter_value().string_value
        debug_overlay_dir = self.get_parameter("debug_overlay_dir").get_parameter_value().string_value
        save_debug_every_n = self.get_parameter("save_debug_every_n").get_parameter_value().integer_value

        self.debug_overlay_dir = Path(debug_overlay_dir).expanduser() if debug_overlay_dir else None
        self.save_debug_every_n = int(save_debug_every_n)
        if self.debug_overlay_dir:
            self.debug_overlay_dir.mkdir(parents=True, exist_ok=True)

        self.runtime = GuqinRealtimeRuntime(
            sam_guqin_dir=sam_guqin_dir,
            checkpoint_path=checkpoint_path,
            threshold=mask_threshold,
            mode=inference_mode,
            expected_strings=int(expected_strings),
            always_recalibrate=bool(always_recalibrate),
            force_recalibrate_every_n=int(force_recalibrate_every_n),
            tracker_max_inlier_dist_px=float(tracker_max_inlier_dist_px),
            tracker_recal_inlier_threshold=float(tracker_recal_inlier_threshold),
        )

        self.mask_pub = self.create_publisher(Image, publish_mask_topic, 10)
        self.overlay_pub = self.create_publisher(Image, publish_overlay_topic, 10)
        self.json_pub = self.create_publisher(String, publish_json_topic, 10)

        # latest-frame-wins: DDS 层只保留最新 1 帧, 不排队旧帧
        latest_only_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._latest_msg: Image | None = None
        self._latest_lock = threading.Lock()
        self._busy = False

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            latest_only_qos,
        )
        # 推理不在图像回调里跑, 由定时器取"当前最新帧"处理,
        # 处理期间到达的帧直接被覆盖丢弃, 延迟 = 单帧推理耗时
        self.process_timer = self.create_timer(0.02, self.process_latest)

        self.get_logger().info(
            f"listening image_topic={image_topic}, checkpoint={checkpoint_path}, "
            f"mode={inference_mode}, always_recalibrate={always_recalibrate}, "
            f"force_recalibrate_every_n={force_recalibrate_every_n}, "
            f"tracker_recal_inlier_threshold={tracker_recal_inlier_threshold:.3f}"
        )

    def _draw_overlay(self, frame_bgr, endpoints):
        overlay = frame_bgr.copy()
        palette = [
            (0, 0, 255),
            (0, 180, 0),
            (0, 215, 255),
            (255, 0, 0),
            (255, 0, 255),
            (255, 255, 0),
            (220, 220, 220),
        ]
        for idx, item in enumerate(endpoints):
            color = palette[idx % len(palette)]
            p0 = tuple(int(round(v)) for v in item["p_start_uv"])
            p1 = tuple(int(round(v)) for v in item["p_end_uv"])
            cv2.line(overlay, p0, p1, color, 2, cv2.LINE_AA)
            cv2.circle(overlay, p0, 4, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p1, 4, color, -1, cv2.LINE_AA)
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            cv2.putText(
                overlay,
                str(item["string_id"]),
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return overlay

    def image_callback(self, msg: Image) -> None:
        # 只覆盖最新帧, 绝不在这里跑模型
        with self._latest_lock:
            self._latest_msg = msg

    def process_latest(self) -> None:
        if self._busy:
            return
        with self._latest_lock:
            msg, self._latest_msg = self._latest_msg, None
        if msg is None:
            return

        self._busy = True
        try:
            self._process_one(msg)
        finally:
            self._busy = False

    def _process_one(self, msg: Image) -> None:
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        t0 = time.monotonic()
        try:
            result = self.runtime.process_frame(frame_bgr)
        except Exception as exc:
            self.get_logger().error(f"runtime failed: {exc}")
            return
        infer_ms = (time.monotonic() - t0) * 1000.0

        if result is None:
            self.get_logger().warn("segmentation foreground too small, skip frame")
            return

        stale_ms = (
            self.get_clock().now().nanoseconds
            - (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        ) / 1e6
        self.get_logger().info(
            f"frame={result.frame_index} seg={result.seg_ms:.0f}ms "
            f"track={result.track_ms:.0f}ms stale={stale_ms:.0f}ms "
            f"roi={int(result.used_roi)} inliers={result.inlier_ratio:.2f} "
            f"shift={result.shift_applied_px:+.0f}px",
            throttle_duration_sec=2.0,
        )

        mask_msg = self.bridge.cv2_to_imgmsg(result.mask, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        # 没人看 overlay 就不画不发, 省下全分辨率绘制+序列化的开销
        if self.overlay_pub.get_subscription_count() > 0:
            overlay = self._draw_overlay(frame_bgr, result.endpoints)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)
        else:
            overlay = None

        payload = self.runtime.result_to_json_dict(result)
        payload["header"] = {
            "frame_id": msg.header.frame_id,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
        }

        json_msg = String()
        json_msg.data = json.dumps(payload, ensure_ascii=False)
        self.json_pub.publish(json_msg)

        if result.calibrated:
            self.get_logger().info(
                f"initial calibration done on frame={result.frame_index}, fg_ratio={result.mask_foreground_ratio:.4f}"
            )
        elif result.recalibrated:
            self.get_logger().warn(
                f"tracker recalibrated on frame={result.frame_index}, inlier_ratio={result.inlier_ratio:.3f}"
            )

        if self.debug_overlay_dir and self.save_debug_every_n > 0:
            if result.frame_index % self.save_debug_every_n == 0:
                if overlay is None:
                    overlay = self._draw_overlay(frame_bgr, result.endpoints)
                out_path = self.debug_overlay_dir / f"frame_{result.frame_index:06d}.jpg"
                cv2.imwrite(str(out_path), overlay)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GuqinStringRealtimeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()