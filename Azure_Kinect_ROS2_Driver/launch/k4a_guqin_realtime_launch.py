from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    k4a_node = Node(
        package="azure_kinect_ros2_driver",
        executable="azure_kinect_node",
        name="k4a_ros2_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "point_cloud": False,
                "rgb_point_cloud": False,
            }
        ],
    )

    guqin_node = Node(
        package="azure_kinect_ros2_driver",
        executable="guqin_string_realtime_node.py",
        name="guqin_string_realtime_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "image_topic": "/k4a/rgb/image_raw",
            }
        ],
    )

    strings_3d_node = Node(
        package="azure_kinect_ros2_driver",
        executable="guqin_strings_3d_node.py",
        name="guqin_strings_3d_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "backend": "depth_image",
                "depth_image_topic": "/k4a/depth_to_rgb/image_raw",
                "camera_info_topic": "/k4a/rgb/camera_info",
                "target_frame": "rgb_camera_link",
                "publish_on_depth": True,
                "publish_rate_hz": 10.0,
            }
        ],
    )

    ld = LaunchDescription()
    ld.add_action(k4a_node)
    ld.add_action(guqin_node)
    ld.add_action(strings_3d_node)
    return ld
