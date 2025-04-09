#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Nodes to launch
    video = Node(
        package="cspe",
        executable="context_identification",
        name="context_identification",
        output="screen",
    )

    mb_simulator = Node(
        package="cspe",
        executable="parameter_estimation",
        name="parameter_estimation",
        output="screen",
    )

    return LaunchDescription([video, mb_simulator])
