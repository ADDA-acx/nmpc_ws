from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():
    package_dir = get_package_share_directory('diff_nmpc_tracker_ros2')
    rviz_config_file = os.path.join(package_dir, 'rviz', 'nmpc_test.rviz')
    return LaunchDescription([
        Node(
            package='diff_nmpc_tracker_ros2',
            executable='diff_nmpc_node',
            name='nmpc_controller',
            output='screen'),
        # Node(
        #     package='diff_nmpc_tracker_ros2',
        #     executable='ref_pub',
        #     name='ref_pub',
        #     output='screen'),
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', rviz_config_file],
        #     output='screen'
        # )

    ])