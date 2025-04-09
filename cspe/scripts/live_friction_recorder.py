import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from context_msgs.msg import ParamList
from context_msgs.msg import ContextStamped
import csv
import os
import time


class FrictionComparisonNode(Node):
    def __init__(self):
        super().__init__("friction_comparison_node")

        # Store values
        self.gt_friction = None
        self.estimated_friction = None
        self.current_cluster = None

        # Create a local data log
        self.log_data = []

        # Subscribers
        self.friction_sub = self.create_subscription(
            Float32, "/friction", self.friction_callback, 10
        )
        self.estimate_sub = self.create_subscription(
            ParamList, "/estimates/current", self.estimate_callback, 10
        )
        self.cluster_sub = self.create_subscription(
            ContextStamped, "/cluster", self.cluster_callback, 10
        )

        # Timer to periodically record data
        self.timer = self.create_timer(0.1, self.record_data_callback)

        self.get_logger().info("FrictionComparisonNode initialized.")

    def friction_callback(self, msg):
        self.gt_friction = msg.data

    def estimate_callback(self, msg):
        # Look for the parameter named "mu" in the ParamList
        for param in msg.params:
            if param.name == "mu":
                self.estimated_friction = param.value
                break

    def cluster_callback(self, msg):
        self.current_cluster = msg.context.context

    def record_data_callback(self):
        # Only record if we have both friction and estimate
        if self.gt_friction is not None and self.estimated_friction is not None:
            error = self.gt_friction - self.estimated_friction
            now = self.get_clock().now().to_msg()
            time_sec = now.sec + now.nanosec * 1e-9
            self.log_data.append(
                {
                    "time": time_sec,
                    "ground_truth": self.gt_friction,
                    "estimate": self.estimated_friction,
                    "error": error,
                    "cluster": self.current_cluster if self.current_cluster else -1,
                }
            )

    def destroy_node(self):
        # On shutdown, save the collected data to a CSV
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        file_name = f"friction_log_{timestamp_str}.csv"
        file_path = os.path.join(os.getcwd(), file_name)

        with open(file_path, "w", newline="") as csvfile:
            fieldnames = ["time", "ground_truth", "estimate", "error", "cluster"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.log_data:
                writer.writerow(row)

        self.get_logger().info(f"Saved friction log data to {file_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FrictionComparisonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
