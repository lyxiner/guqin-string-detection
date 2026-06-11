#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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
            "sliding",
            ParameterDescriptor(description="resize or sliding"),
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
        )

        self.mask_pub = self.create_publisher(Image, publish_mask_topic, 10)
        self.overlay_pub = self.create_publisher(Image, publish_overlay_topic, 10)
        self.json_pub = self.create_publisher(String, publish_json_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"listening image_topic={image_topic}, checkpoint={checkpoint_path}, mode={inference_mode}"
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
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        try:
            result = self.runtime.process_frame(frame_bgr)
        except Exception as exc:
            self.get_logger().error(f"runtime failed: {exc}")
            return

        if result is None:
            self.get_logger().warn("segmentation foreground too small, skip frame")
            return

        mask_msg = self.bridge.cv2_to_imgmsg(result.mask, encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        overlay = self._draw_overlay(frame_bgr, result.endpoints)
        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

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
