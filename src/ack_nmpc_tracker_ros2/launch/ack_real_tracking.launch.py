from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():
    package_dir = get_package_share_directory('ack_nmpc_tracker_ros2')
    rviz_config_file = os.path.join(package_dir, 'rviz', 'nmpc_test.rviz')
    return LaunchDescription([
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', rviz_config_file],
        #     output='screen'
        # ),
        # Node(
        #     package='ack_nmpc_tracker_ros2',
        #     executable='ack_model_node',
        #     name='unicycle_sim',
        #     parameters=[{'init_pose': [0.0, 0.0, 0.0]}],
        #     output='screen'),
        Node(
            package='ack_nmpc_tracker_ros2',
            executable='ack_nmpc_node',
            name='nmpc_controller',
            output='screen'),
        # Node(
        #     package='ack_nmpc_tracker_ros2',
        #     executable='ref_pub',
        #     name='ref_pub',
        #     output='screen'),
        # Node(
        #     package='ack_nmpc_tracker_ros2',
        #     executable='cmd_vel_to_overlay',
        #     name='cmd_vel_to_overlay',
        #     output='screen')
    ])