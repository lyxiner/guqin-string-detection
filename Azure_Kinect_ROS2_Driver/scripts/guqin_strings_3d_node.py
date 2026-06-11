#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import struct
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, String
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray


FLOAT32_FIELD = PointField.FLOAT32


@dataclass(frozen=True)
class CloudAccessor:
    msg: PointCloud2
    x_offset: int
    y_offset: int
    z_offset: int
    fmt: str

    @classmethod
    def from_msg(cls, msg: PointCloud2) -> "CloudAccessor":
        offsets: dict[str, int] = {}
        datatypes: dict[str, int] = {}
        for field in msg.fields:
            offsets[field.name] = field.offset
            datatypes[field.name] = field.datatype

        missing = [name for name in ("x", "y", "z") if name not in offsets]
        if missing:
            raise ValueError(f"PointCloud2 missing fields: {missing}")
        unsupported = [
            name for name in ("x", "y", "z") if datatypes[name] != FLOAT32_FIELD
        ]
        if unsupported:
            raise ValueError(f"PointCloud2 fields are not float32: {unsupported}")

        return cls(
            msg=msg,
            x_offset=offsets["x"],
            y_offset=offsets["y"],
            z_offset=offsets["z"],
            fmt=">f" if msg.is_bigendian else "<f",
        )

    def point_at(self, u: int, v: int) -> np.ndarray | None:
        if u < 0 or v < 0 or u >= self.msg.width or v >= self.msg.height:
            return None

        base = v * self.msg.row_step + u * self.msg.point_step
        data = self.msg.data
        x = struct.unpack_from(self.fmt, data, base + self.x_offset)[0]
        y = struct.unpack_from(self.fmt, data, base + self.y_offset)[0]
        z = struct.unpack_from(self.fmt, data, base + self.z_offset)[0]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        return np.array([x, y, z], dtype=np.float64)


@dataclass(frozen=True)
class DepthFrame:
    image_m: np.ndarray
    header: Any


def quat_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    rotation = quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
    translation = np.array([t.x, t.y, t.z], dtype=np.float64)
    return (rotation @ points.T).T + translation


def make_point(values: np.ndarray | list[float]) -> Point:
    p = Point()
    p.x = float(values[0])
    p.y = float(values[1])
    p.z = float(values[2])
    return p


class GuqinStrings3DNode(Node):
    def __init__(self) -> None:
        super().__init__("guqin_strings_3d_node")

        self.declare_parameter(
            "backend",
            "depth_image",
            ParameterDescriptor(description="3D backend: depth_image or point_cloud"),
        )
        self.declare_parameter(
            "fit_json_topic",
            "/guqin/strings_fit_json",
            ParameterDescriptor(description="2D string fit JSON topic"),
        )
        self.declare_parameter(
            "depth_image_topic",
            "/k4a/depth_to_rgb/image_raw",
            ParameterDescriptor(description="Depth image aligned to RGB pixels"),
        )
        self.declare_parameter(
            "camera_info_topic",
            "/k4a/rgb/camera_info",
            ParameterDescriptor(description="RGB camera intrinsics for depth_image backend"),
        )
        self.declare_parameter(
            "point_cloud_topic",
            "/k4a/points2",
            ParameterDescriptor(description="Organized RGB-frame PointCloud2 topic"),
        )
        self.declare_parameter(
            "target_frame",
            "rgb_camera_link",
            ParameterDescriptor(description="Target frame for published 3D strings"),
        )
        self.declare_parameter(
            "publish_json_topic",
            "/guqin/strings_3d_json",
            ParameterDescriptor(description="Output 3D string JSON topic"),
        )
        self.declare_parameter(
            "publish_marker_topic",
            "/guqin/strings_3d_markers",
            ParameterDescriptor(description="Output RViz marker topic"),
        )
        self.declare_parameter(
            "samples_per_string",
            21,
            ParameterDescriptor(description="Number of samples along each 2D string"),
        )
        self.declare_parameter(
            "trim_endpoint_ratio",
            0.05,
            ParameterDescriptor(description="Fraction trimmed from both endpoints while sampling"),
        )
        self.declare_parameter(
            "search_radius_px",
            3,
            ParameterDescriptor(description="Pixel radius used to recover nearby valid depth samples"),
        )
        self.declare_parameter(
            "min_valid_samples",
            5,
            ParameterDescriptor(description="Minimum valid 3D samples required for a string"),
        )
        self.declare_parameter(
            "outlier_threshold_m",
            0.02,
            ParameterDescriptor(description="3D line RANSAC inlier threshold"),
        )
        self.declare_parameter(
            "max_line_rms_m",
            0.02,
            ParameterDescriptor(description="Reject strings with final 3D line RMS above this"),
        )
        self.declare_parameter(
            "min_depth_m",
            0.05,
            ParameterDescriptor(description="Reject point-cloud samples closer than this"),
        )
        self.declare_parameter(
            "max_depth_m",
            3.0,
            ParameterDescriptor(description="Reject point-cloud samples farther than this"),
        )
        self.declare_parameter(
            "tf_timeout_sec",
            1.0,
            ParameterDescriptor(description="TF lookup timeout"),
        )
        self.declare_parameter(
            "max_cloud_age_sec",
            1.0,
            ParameterDescriptor(description="Warn when latest 3D source is older than this"),
        )
        self.declare_parameter(
            "publish_on_depth",
            True,
            ParameterDescriptor(
                description="For depth_image backend, publish 3D strings from depth callbacks using latest 2D fit"
            ),
        )
        self.declare_parameter(
            "publish_rate_hz",
            10.0,
            ParameterDescriptor(description="Maximum 3D publish rate for publish_on_depth, <=0 means unlimited"),
        )
        self.declare_parameter(
            "max_fit_age_sec",
            2.0,
            ParameterDescriptor(description="Reject latest 2D fit if it is older than this"),
        )

        self.backend = self._string_param("backend")
        if self.backend not in {"depth_image", "point_cloud"}:
            raise ValueError("backend must be 'depth_image' or 'point_cloud'")

        self.fit_json_topic = self._string_param("fit_json_topic")
        self.depth_image_topic = self._string_param("depth_image_topic")
        self.camera_info_topic = self._string_param("camera_info_topic")
        self.point_cloud_topic = self._string_param("point_cloud_topic")
        self.target_frame = self._string_param("target_frame")
        self.samples_per_string = max(2, self._int_param("samples_per_string"))
        self.trim_endpoint_ratio = float(
            np.clip(self._double_param("trim_endpoint_ratio"), 0.0, 0.45)
        )
        self.search_radius_px = max(0, self._int_param("search_radius_px"))
        self.min_valid_samples = max(2, self._int_param("min_valid_samples"))
        self.outlier_threshold_m = self._double_param("outlier_threshold_m")
        self.max_line_rms_m = self._double_param("max_line_rms_m")
        self.min_depth_m = self._double_param("min_depth_m")
        self.max_depth_m = self._double_param("max_depth_m")
        self.tf_timeout = self._double_param("tf_timeout_sec")
        self.max_source_age = self._double_param("max_cloud_age_sec")
        self.publish_on_depth = self._bool_param("publish_on_depth")
        self.publish_rate_hz = self._double_param("publish_rate_hz")
        self.max_fit_age = self._double_param("max_fit_age_sec")

        self.bridge = CvBridge()
        self._fit_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._info_lock = threading.Lock()
        self._cloud_lock = threading.Lock()
        self._latest_fit_data: dict[str, Any] | None = None
        self._latest_depth: DepthFrame | None = None
        self._camera_info: CameraInfo | None = None
        self._latest_cloud: PointCloud2 | None = None
        self._last_publish_time_ns = 0
        self._logged_depth_shape = False
        self._logged_cloud_shape = False
        self._last_width_warning = ""

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.json_pub = self.create_publisher(
            String, self._string_param("publish_json_topic"), 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, self._string_param("publish_marker_topic"), 10
        )

        if self.backend == "depth_image":
            self.create_subscription(
                Image,
                self.depth_image_topic,
                self._depth_cb,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_cb,
                qos_profile_sensor_data,
            )
        else:
            self.create_subscription(
                PointCloud2,
                self.point_cloud_topic,
                self._cloud_cb,
                qos_profile_sensor_data,
            )
        self.create_subscription(String, self.fit_json_topic, self._json_cb, 10)

        if self.backend == "depth_image":
            self.get_logger().info(
                "listening fit_json_topic=%s, depth_image_topic=%s, camera_info_topic=%s, "
                "target_frame=%s"
                % (
                    self.fit_json_topic,
                    self.depth_image_topic,
                    self.camera_info_topic,
                    self.target_frame,
                )
            )
            self.get_logger().info(
                "depth_image backend is active; /k4a/points2 is not required."
            )
        else:
            self.get_logger().info(
                "listening fit_json_topic=%s, point_cloud_topic=%s, target_frame=%s"
                % (self.fit_json_topic, self.point_cloud_topic, self.target_frame)
            )
            self.get_logger().info(
                "Kinect launch must use rgb_point_cloud=true and point_cloud_in_depth_frame=false "
                "so /k4a/points2 is organized in RGB pixels."
            )

    def _string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _int_param(self, name: str) -> int:
        return int(self.get_parameter(name).get_parameter_value().integer_value)

    def _double_param(self, name: str) -> float:
        return float(self.get_parameter(name).get_parameter_value().double_value)

    def _bool_param(self, name: str) -> bool:
        return bool(self.get_parameter(name).get_parameter_value().bool_value)

    def _depth_cb(self, msg: Image) -> None:
        fit_data_for_publish: dict[str, Any] | None = None
        if self.backend == "depth_image" and self.publish_on_depth:
            fit_data_for_publish = self._reserve_depth_publish()
            if fit_data_for_publish is None:
                return

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warn(f"failed to convert depth image: {exc}")
            return

        if image.dtype == np.uint16:
            depth_m = image.astype(np.float32) / 1000.0
        elif image.dtype == np.float32:
            depth_m = image.copy()
        elif image.dtype == np.float64:
            depth_m = image.astype(np.float32)
        else:
            self.get_logger().warn(
                f"unexpected depth image dtype={image.dtype}; converting to float32"
            )
            depth_m = image.astype(np.float32)

        with self._depth_lock:
            self._latest_depth = DepthFrame(image_m=depth_m, header=msg.header)

        if not self._logged_depth_shape:
            self._logged_depth_shape = True
            valid = depth_m[np.isfinite(depth_m) & (depth_m > 0.0)]
            if valid.size:
                self.get_logger().info(
                    "received depth image: frame=%s shape=%s dtype=%s valid=%.3f..%.3fm"
                    % (
                        msg.header.frame_id,
                        depth_m.shape,
                        image.dtype,
                        float(valid.min()),
                        float(valid.max()),
                    )
                )
            else:
                self.get_logger().warn(
                    "received depth image: frame=%s shape=%s but no valid depth"
                    % (msg.header.frame_id, depth_m.shape)
                )

        if self.backend == "depth_image" and self.publish_on_depth:
            self._publish_from_depth_fit(fit_data_for_publish)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        with self._info_lock:
            self._camera_info = msg

    def _cloud_cb(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_cloud = msg

        if not self._logged_cloud_shape:
            self._logged_cloud_shape = True
            self.get_logger().info(
                "received organized cloud: frame=%s width=%d height=%d point_step=%d"
                % (msg.header.frame_id, msg.width, msg.height, msg.point_step)
            )

    def _json_cb(self, msg: String) -> None:
        try:
            fit_data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"failed to parse fit JSON: {exc}")
            return

        with self._fit_lock:
            self._latest_fit_data = fit_data

        if self.backend == "depth_image" and self.publish_on_depth:
            return

        try:
            if self.backend == "depth_image":
                output = self._resolve_strings_from_depth(fit_data=fit_data)
            else:
                output = self._resolve_strings_from_cloud(fit_data)
        except Exception as exc:
            self.get_logger().warn(f"3D string resolve failed: {exc}")
            return

        self._publish_output(output)

    def _reserve_depth_publish(self) -> dict[str, Any] | None:
        now_ns = self.get_clock().now().nanoseconds
        if self.publish_rate_hz > 0.0:
            min_period_ns = int(1e9 / self.publish_rate_hz)
            if now_ns - self._last_publish_time_ns < min_period_ns:
                return None

        with self._info_lock:
            info_ready = self._camera_info is not None
        if not info_ready:
            return None

        with self._fit_lock:
            fit_data = self._latest_fit_data
        if fit_data is None:
            return None

        fit_age = self._fit_age_sec(fit_data)
        if fit_age is not None and fit_age > self.max_fit_age:
            self._last_publish_time_ns = now_ns
            self.get_logger().warn(
                f"latest 2D fit is {fit_age:.2f}s old; skip depth-driven 3D publish"
            )
            return None

        self._last_publish_time_ns = now_ns
        return fit_data

    def _publish_from_depth_fit(self, fit_data: dict[str, Any]) -> None:
        try:
            output = self._resolve_strings_from_depth(fit_data=fit_data)
        except Exception as exc:
            self.get_logger().warn(f"depth-driven 3D string resolve failed: {exc}")
            return

        self._publish_output(output)

    def _publish_output(self, output: dict[str, Any]) -> bool:
        if not output["strings"]:
            return False

        out_msg = String()
        out_msg.data = json.dumps(output, ensure_ascii=False)
        self.json_pub.publish(out_msg)
        self.marker_pub.publish(self._make_markers(output))
        return True

    def _stamp_age_sec(self, stamp_msg) -> float | None:
        if stamp_msg.sec == 0 and stamp_msg.nanosec == 0:
            return None
        return max(0.0, self._now_sec() - self._stamp_to_sec(stamp_msg))

    def _fit_age_sec(self, fit_data: dict[str, Any]) -> float | None:
        stamp = fit_data.get("header", {}).get("stamp", {})
        sec = int(stamp.get("sec", 0))
        nanosec = int(stamp.get("nanosec", 0))
        if sec == 0 and nanosec == 0:
            return None
        return max(0.0, self._now_sec() - (float(sec) + float(nanosec) * 1e-9))

    def _now_sec(self) -> float:
        now = self.get_clock().now().to_msg()
        return self._stamp_to_sec(now)

    def _stamp_to_sec(self, stamp_msg) -> float:
        return float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9

    def _resolve_strings_from_cloud(self, fit_data: dict[str, Any]) -> dict[str, Any]:
        with self._cloud_lock:
            cloud = self._latest_cloud

        if cloud is None:
            raise ValueError("no point cloud received yet")

        source_age = self._stamp_age_sec(cloud.header.stamp)
        if source_age is not None and source_age > self.max_source_age:
            self.get_logger().warn(
                f"latest point cloud is {source_age:.2f}s old; check /k4a/points2"
            )

        if cloud.height <= 1:
            raise ValueError(
                "point cloud is unorganized (height <= 1). Rebuild/restart azure_kinect_node "
                "so /k4a/points2 keeps RGB image width/height."
            )

        accessor = CloudAccessor.from_msg(cloud)
        return self._resolve_strings_common(
            fit_data=fit_data,
            source_frame_id=cloud.header.frame_id,
            source_stamp=cloud.header.stamp,
            source_width=int(cloud.width),
            source_height=int(cloud.height),
            sample_point_fn=lambda u, v: self._sample_nearest_valid_cloud(
                accessor, u, v
            ),
        )

    def _resolve_strings_from_depth(self, fit_data: dict[str, Any]) -> dict[str, Any]:
        with self._depth_lock:
            depth = self._latest_depth
        with self._info_lock:
            info = self._camera_info

        if depth is None:
            raise ValueError("no aligned depth image received yet")
        if info is None:
            raise ValueError("no RGB camera_info received yet")

        source_age = self._stamp_age_sec(depth.header.stamp)
        if source_age is not None and source_age > self.max_source_age:
            self.get_logger().warn(
                f"latest depth image is {source_age:.2f}s old; check {self.depth_image_topic}"
            )

        height, width = depth.image_m.shape[:2]
        k = info.k
        fx, fy = float(k[0]), float(k[4])
        cx, cy = float(k[2]), float(k[5])
        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise ValueError("invalid camera_info intrinsics")

        return self._resolve_strings_common(
            fit_data=fit_data,
            source_frame_id=depth.header.frame_id or info.header.frame_id,
            source_stamp=depth.header.stamp,
            source_width=int(width),
            source_height=int(height),
            sample_point_fn=lambda u, v: self._sample_nearest_valid_depth(
                depth.image_m, u, v, fx, fy, cx, cy
            ),
        )

    def _resolve_strings_common(
        self,
        fit_data: dict[str, Any],
        source_frame_id: str,
        source_stamp,
        source_width: int,
        source_height: int,
        sample_point_fn,
    ) -> dict[str, Any]:
        endpoints = self._extract_endpoints(fit_data)
        if not endpoints:
            raise ValueError("fit JSON does not contain endpoints/strings")

        image_size = self._extract_image_size(fit_data)
        if image_size:
            width, height = image_size
            if (width, height) != (source_width, source_height):
                warning_key = f"{width}x{height}->{source_width}x{source_height}"
                if warning_key != self._last_width_warning:
                    self._last_width_warning = warning_key
                    self.get_logger().warn(
                        "2D fit image size is %dx%d but 3D source is %dx%d."
                        % (width, height, source_width, source_height)
                    )

        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame_id,
            rclpy.time.Time(),
            Duration(seconds=self.tf_timeout),
        )

        strings = []
        for item in endpoints:
            resolved = self._resolve_one_string(item, sample_point_fn, transform)
            if resolved is not None:
                strings.append(resolved)

        strings.sort(key=lambda value: value["string_id"])
        now_msg = self.get_clock().now().to_msg()
        return {
            "header": {
                "frame_id": self.target_frame,
                "stamp": {
                    "sec": int(now_msg.sec),
                    "nanosec": int(now_msg.nanosec),
                },
            },
            "source_header": {
                "frame_id": source_frame_id,
                "stamp": {
                    "sec": int(source_stamp.sec),
                    "nanosec": int(source_stamp.nanosec),
                },
            },
            "source_fit_header": fit_data.get("header", {}),
            "n_strings": len(strings),
            "strings": strings,
        }

    def _extract_endpoints(self, fit_data: dict[str, Any]) -> list[dict[str, Any]]:
        endpoints = fit_data.get("endpoints")
        if endpoints:
            return list(endpoints)
        strings = fit_data.get("strings")
        if strings:
            return list(strings)
        return []

    def _extract_image_size(self, fit_data: dict[str, Any]) -> tuple[int, int] | None:
        image_size = fit_data.get("image_size_uv")
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
            if width and height:
                return int(width), int(height)
        return None

    def _resolve_one_string(
        self,
        item: dict[str, Any],
        sample_point_fn,
        transform,
    ) -> dict[str, Any] | None:
        sid = int(item["string_id"])
        p0 = np.asarray(item["p_start_uv"], dtype=np.float64)
        p1 = np.asarray(item["p_end_uv"], dtype=np.float64)
        ts = np.linspace(
            self.trim_endpoint_ratio,
            1.0 - self.trim_endpoint_ratio,
            self.samples_per_string,
        )

        points_cam: list[np.ndarray] = []
        valid_ts: list[float] = []
        for t in ts:
            uv = p0 + t * (p1 - p0)
            point = sample_point_fn(int(round(uv[0])), int(round(uv[1])))
            if point is None:
                continue
            points_cam.append(point)
            valid_ts.append(float(t))

        if len(points_cam) < self.min_valid_samples:
            self.get_logger().warn(
                f"string {sid}: only {len(points_cam)}/{len(ts)} valid 3D samples"
            )
            return None

        points_target = transform_points(np.vstack(points_cam), transform)
        valid_ts_array = np.asarray(valid_ts, dtype=np.float64)
        try:
            inlier_mask = self._ransac_line_inliers(
                points_target,
                threshold_m=self.outlier_threshold_m,
            )
            n_inliers = int(inlier_mask.sum())
            if n_inliers < self.min_valid_samples:
                self.get_logger().warn(
                    f"string {sid}: only {n_inliers}/{len(points_cam)} inlier 3D samples"
                )
                return None

            line = self._fit_parametric_line(
                points_target[inlier_mask],
                valid_ts_array[inlier_mask],
            )
        except ValueError as exc:
            self.get_logger().warn(f"string {sid}: {exc}")
            return None

        valid_ratio = len(points_cam) / len(ts)
        if line["rms_m"] > self.max_line_rms_m:
            self.get_logger().warn(
                f"string {sid}: reject 3D line residual {line['rms_m'] * 1000.0:.1f}mm "
                f"> {self.max_line_rms_m * 1000.0:.1f}mm"
            )
            return None

        return {
            "string_id": sid,
            "frame_id": self.target_frame,
            "p_start": line["p_start"].tolist(),
            "p_end": line["p_end"].tolist(),
            "p_mid": line["p_mid"].tolist(),
            "direction_unit": line["direction_unit"].tolist(),
            "length_m": float(np.linalg.norm(line["p_end"] - line["p_start"])),
            "valid_samples": len(points_cam),
            "inlier_samples": n_inliers,
            "rejected_samples": len(points_cam) - n_inliers,
            "total_samples": len(ts),
            "valid_sample_ratio": float(valid_ratio),
            "sample_t_min": float(valid_ts_array.min()),
            "sample_t_max": float(valid_ts_array.max()),
            "line_rms_m": line["rms_m"],
            "source_uv": {
                "p_start_uv": p0.tolist(),
                "p_end_uv": p1.tolist(),
            },
        }

    def _sample_nearest_valid_cloud(
        self,
        accessor: CloudAccessor,
        u: int,
        v: int,
    ) -> np.ndarray | None:
        for du, dv in self._offsets_by_radius(self.search_radius_px):
            point = accessor.point_at(u + du, v + dv)
            if point is None:
                continue
            depth = float(point[2])
            if self.min_depth_m <= depth <= self.max_depth_m:
                return point
        return None

    def _sample_nearest_valid_depth(
        self,
        depth_m: np.ndarray,
        u: int,
        v: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray | None:
        height, width = depth_m.shape[:2]
        for du, dv in self._offsets_by_radius(self.search_radius_px):
            uu = u + du
            vv = v + dv
            if uu < 0 or vv < 0 or uu >= width or vv >= height:
                continue
            depth = float(depth_m[vv, uu])
            if not math.isfinite(depth):
                continue
            if self.min_depth_m <= depth <= self.max_depth_m:
                x = (float(uu) - cx) * depth / fx
                y = (float(vv) - cy) * depth / fy
                z = depth
                return np.array([x, y, z], dtype=np.float64)
        return None

    def _offsets_by_radius(self, radius: int) -> list[tuple[int, int]]:
        offsets = []
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                offsets.append((du, dv))
        offsets.sort(key=lambda value: value[0] * value[0] + value[1] * value[1])
        return offsets

    def _ransac_line_inliers(self, points: np.ndarray, threshold_m: float) -> np.ndarray:
        if len(points) < 2:
            raise ValueError("not enough points for 3D line RANSAC")

        best_mask: np.ndarray | None = None
        best_count = 0
        best_score = float("inf")
        threshold_m = max(float(threshold_m), 1e-6)

        for i in range(len(points) - 1):
            p0 = points[i]
            for j in range(i + 1, len(points)):
                direction = points[j] - p0
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    continue
                direction /= norm
                projected = p0 + np.outer((points - p0) @ direction, direction)
                residuals = np.linalg.norm(points - projected, axis=1)
                mask = residuals <= threshold_m
                count = int(mask.sum())
                if count == 0:
                    continue
                score = float(np.median(residuals[mask]))
                if count > best_count or (count == best_count and score < best_score):
                    best_mask = mask
                    best_count = count
                    best_score = score

        if best_mask is None:
            raise ValueError("3D line RANSAC failed")
        return best_mask

    def _fit_parametric_line(
        self, points: np.ndarray, ts: np.ndarray
    ) -> dict[str, np.ndarray | float]:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        if singular_values[0] < 1e-9:
            raise ValueError("3D samples are degenerate")

        direction = vt[0]
        signed_dist = centered @ direction
        slope, intercept = np.linalg.lstsq(
            np.vstack([ts, np.ones_like(ts)]).T,
            signed_dist,
            rcond=None,
        )[0]
        if slope < 0.0:
            direction = -direction
            signed_dist = -signed_dist
            slope = -slope
            intercept = -intercept

        projected = centroid + np.outer(signed_dist, direction)
        residual = points - projected
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))

        def point_at(t: float) -> np.ndarray:
            return centroid + (slope * t + intercept) * direction

        return {
            "p_start": point_at(0.0),
            "p_end": point_at(1.0),
            "p_mid": point_at(0.5),
            "direction_unit": direction / np.linalg.norm(direction),
            "rms_m": rms,
        }

    def _make_markers(self, output: dict[str, Any]) -> MarkerArray:
        markers = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        line_marker = Marker()
        line_marker.header.frame_id = output["header"]["frame_id"]
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "guqin_strings_3d"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.004
        line_marker.color = ColorRGBA(r=0.1, g=0.8, b=1.0, a=1.0)

        midpoint_marker = Marker()
        midpoint_marker.header = line_marker.header
        midpoint_marker.ns = "guqin_string_midpoints"
        midpoint_marker.id = 1
        midpoint_marker.type = Marker.SPHERE_LIST
        midpoint_marker.action = Marker.ADD
        midpoint_marker.scale.x = 0.015
        midpoint_marker.scale.y = 0.015
        midpoint_marker.scale.z = 0.015
        midpoint_marker.color = ColorRGBA(r=1.0, g=0.6, b=0.1, a=1.0)

        for idx, string in enumerate(output["strings"], start=1):
            line_marker.points.append(make_point(string["p_start"]))
            line_marker.points.append(make_point(string["p_end"]))
            midpoint_marker.points.append(make_point(string["p_mid"]))

            text = Marker()
            text.header = line_marker.header
            text.ns = "guqin_string_ids"
            text.id = 100 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = make_point(string["p_mid"])
            text.pose.position.z += 0.025
            text.pose.orientation.w = 1.0
            text.scale.z = 0.035
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = str(string["string_id"])
            markers.markers.append(text)

        markers.markers.append(line_marker)
        markers.markers.append(midpoint_marker)
        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GuqinStrings3DNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
