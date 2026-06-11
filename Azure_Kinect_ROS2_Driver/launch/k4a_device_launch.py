from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()
    k4a_node = Node(
        package="azure_kinect_ros2_driver",
        executable="azure_kinect_node",
        name="k4a_ros2_node",
        output="screen",
        emulate_tty=True,
        parameters=[{
            'point_cloud': True,
            'rgb_point_cloud': True,
            'point_cloud_in_depth_frame': False,
        }]
    )
    ld.add_action(k4a_node)
    return ld