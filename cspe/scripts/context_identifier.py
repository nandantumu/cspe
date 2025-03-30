import os

import cv2
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from context_msgs.msg import (
    ContextDuration,
    ContextDurationMap,
    ContextStamped,
    Duration,
    DurationList,
)
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage

from cspe.identification.context_id import ContextIdentifier
from cspe.utilities import s_to_s_and_ns


class ContextIdentificationNode(Node):
    def __init__(self):
        super().__init__("terrain_context_identification")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.root_path = get_package_share_directory("cspe")
        self.get_logger().info(f"Root path: {self.root_path}")
        self.csv_path = self.root_path + "/data/cluster"
        self.get_logger().info(f"Saving in: {self.csv_path}")

        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        self.image_listener_exec_group = MutuallyExclusiveCallbackGroup()
        self.clustering_exec_group = MutuallyExclusiveCallbackGroup()
        self.clustering_process_exec_group = MutuallyExclusiveCallbackGroup()

        self.context_id = ContextIdentifier(max_database_size=10000)
        self.get_logger().info("Context Identifier Initialized")

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            ROSImage,
            "/camera/image_raw",
            self.image_callback,
            10,
            callback_group=self.image_listener_exec_group,
        )

        self.cluster_publisher = self.create_publisher(ContextStamped, "/cluster", 10)
        self.cluster_duration_publisher = self.create_publisher(
            ContextDurationMap, "/cluster_durations", 10
        )

        self.image_processing_timer = self.create_timer(
            1.0, self.cluster_callback, callback_group=self.clustering_exec_group
        )

        self.clustering_timer = self.create_timer(
            5.0,
            self.reprocess_cluster_callback,
            callback_group=self.clustering_process_exec_group,
        )

    def get_timestamp(self, msg):
        """Extracts and returns the timestamp as float (sec + nanosec)."""

        return float(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting from ROS Image to OpenCV: {e}")

        timestamp = self.get_timestamp(msg)
        self.context_id.add_to_queue(cv_image, timestamp)

    def cluster_callback(self):
        """
        This function does the following:
        1. Processes the image queue
        2. Gets a fast cluster estimate for the current batch
        3. Publishes the current cluster estimate
        """
        print("Queue Size: ", len(self.context_id.batch_queue))
        if len(self.context_id.batch_queue) == 0:
            return
        # Process the image queue
        cluster_estimates = self.context_id.process_queue()
        try:
            current_cluster = cluster_estimates[-1]
        except IndexError:
            current_cluster = -1

        # Publish the current cluster
        current_msg = ContextStamped()
        current_msg.header.stamp = self.get_clock().now().to_msg()
        current_msg.header.frame_id = "robot"
        current_msg.context.context = int(current_cluster)
        self.cluster_publisher.publish(current_msg)

        # TODO: Publish the time durations of the clusters
        print("Size of Image DB: ", self.context_id.image_db.size())
        print("Current Cluster: ", current_cluster)
        self.context_id.image_db.save_to_disk(self.csv_path + "/imagedb.pth")

    def reprocess_cluster_callback(self):
        """
        This function does the following:
        1. Reprocesses the clusters
        2. Publishes the time durations of the clusters
        """
        print("Reprocessing Clusters")
        current_cluster = self.context_id.reprocess_clusters()
        # Publish the cluster durations
        cluster_durations = self.context_id.get_cluster_durations()
        durations_list = []
        for key in cluster_durations.keys():
            key_durations = []
            for duration in cluster_durations[key]:
                s, ns = s_to_s_and_ns(duration[0])
                start_time = Time(sec=s, nanosec=ns)
                s, ns = s_to_s_and_ns(duration[1])
                end_time = Time(sec=s, nanosec=ns)
                key_durations.append(Duration(start=start_time, end=end_time))
            durations_list.append(
                ContextDuration(
                    key=int(key), durations=DurationList(durations=key_durations)
                )
            )
        cluster_duration_msg = ContextDurationMap(
            pairs=durations_list,
        )
        cluster_duration_msg.header.stamp = self.get_clock().now().to_msg()
        cluster_duration_msg.header.frame_id = "robot"
        self.cluster_duration_publisher.publish(cluster_duration_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ContextIdentificationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
