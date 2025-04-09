import os
from copy import deepcopy

import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from context_msgs.msg import (
    ContextDuration,
    ContextDurationMap,
    ContextParamMap,
    ContextParams,
    ContextStamped,
    Duration,
    DurationList,
    ParamList,
    ParamValue,
    STCombined,
)
from geometry_msgs.msg import PoseWithCovarianceStamped
from pit.dynamics.single_track import SingleTrack
from pit.integration import RK4
from pit.parameters import PointParameterGroup
from pit.utilities.data import (
    create_batched_track_states_and_controls,
    create_single_track_states_and_controls,
)
from pit.utilities.training import find_near_optimal_mu, gradient_search_for_mu
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    DurabilityPolicy,
    LivelinessPolicy,
)
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Imu
import numpy as np

from ..utilities import StateDB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParameterEstimator(Node):
    def __init__(self):
        super().__init__("parameter_estimator")
        self.root_path = get_package_share_directory("cspe")
        self.get_logger().info(f"Root path: {self.root_path}")

        self.cluster_friction_publisher = self.create_publisher(
            ContextParamMap, "/estimates/complete", 10
        )
        self.current_friction_publisher = self.create_publisher(
            ParamList, "/estimates/current", 10
        )

        self.current_context_cbg = MutuallyExclusiveCallbackGroup()
        self.state_listener_cbg = MutuallyExclusiveCallbackGroup()
        self.context_processing_cbg = MutuallyExclusiveCallbackGroup()

        # self.state_qos = QoSProfile(
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT,
        #     history=QoSHistoryPolicy.KEEP_LAST,
        #     durability=DurabilityPolicy.VOLATILE,
        #     liveliness=LivelinessPolicy.AUTOMATIC,
        #     depth=1,
        # )

        self.context_sub = self.create_subscription(
            ContextStamped,
            "/cluster",
            self.current_context_callback,
            10,
            callback_group=self.current_context_cbg,
        )
        self.context_duration_sub = self.create_subscription(
            ContextDurationMap,
            "/cluster_durations",
            self.cluster_duration_callback,
            10,
            callback_group=self.context_processing_cbg,
        )

        self.gt_info_sub = self.create_subscription(
            STCombined,
            "/ground_truth/combined",
            self.gt_info_callback,
            10,
            callback_group=self.state_listener_cbg,
        )

        self.state_db = StateDB(max_size=1_000_000)

        self.params_list = [
            "mu",
            # "Csf",
            # "Csr",
        ]
        self.st_params_list = [
            "l",
            "m",
            "Iz",
            "lf",
            "lr",
            "hcg",
            "Csf",
            "Csr",
            "mu",
        ]
        self.default_params = {
            "l": 4.298,
            "m": 1.225,
            "Iz": 1.538,
            "lf": 0.883,
            "lr": 1.508,
            "hcg": 0.557,
            "Csf": 20.89,
            "Csr": 20.89,
            "mu": 1.048,
        }

        self.context_params = {
            -1: deepcopy(self.default_params),
        }

        self.durations = dict()
        # For each cluster_id, we have a list of durations. Each duration is a list of tuples (start, end).

        self.get_logger().info("Parameter Estimator Initialized")

    def gt_info_callback(self, msg):
        """
        This method is called when the ground truth information is received. It stores the information in the state database.
        """
        # Store the ground truth information in the state database
        self.state_db.add(
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            x=msg.state.x,
            y=msg.state.y,
            velocity=msg.state.velocity,
            yaw=msg.state.yaw,
            yaw_rate=msg.state.yaw_rate,
            slip_angle=msg.state.slip_angle,
            steering_angle=msg.control.steering_angle,
            acceleration=msg.control.acceleration,
        )

    def current_context_callback(self, msg):
        """Based on the current context id, this method looks up the parameter list for this context id and publishes it."""
        context_id = msg.context.context
        try:
            params = self.context_params[context_id]
        except KeyError:
            # If the context id is not found, use the default parameters
            params = self.context_params[-1]
        # Create a ParamList message
        param_list = ParamList()
        param_list.params = []
        # Populate the ParamList message with the parameters
        for param in self.params_list:
            param_value = ParamValue()
            param_value.name = param
            param_value.value = params[param]
            param_list.params.append(param_value)
        # Publish the ParamList message
        self.current_friction_publisher.publish(param_list)
        self.get_logger().info(f"Published parameters for context {context_id}")

    def create_data_dict_from_durations(self, duration):
        """
        This method creates a dictionary from the duration list. The keys are the start times and the values are the end times.
        """
        db_slice = self.state_db.get_index_slice_by_duration(
            start=duration[0], end=duration[1]
        )
        data_dict = {
            "x": [],
            "y": [],
            "velocity": [],
            "yaw": [],
            "yaw_rate": [],
            "slip_angle": [],
            "steering_angle": [],
            "acceleration": [],
        }
        db_rows = self.state_db[db_slice]
        for row in db_rows:
            data_dict["x"].append(row["x"])
            data_dict["y"].append(row["y"])
            data_dict["velocity"].append(row["velocity"])
            data_dict["yaw"].append(row["yaw"])
            data_dict["yaw_rate"].append(row["yaw_rate"])
            data_dict["slip_angle"].append(row["slip_angle"])
            data_dict["steering_angle"].append(row["steering_angle"])
            data_dict["acceleration"].append(row["acceleration"])

        # Calculate the delta time
        dt = []
        for i in range(len(data_dict["x"]) - 1):
            dt.append(db_rows[i + 1]["timestamp"] - db_rows[i]["timestamp"])
        data_dict["dt"] = dt
        # Add the last delta time
        try:
            next_time = db_slice.stop
            last_item = self.state_db[next_time]
            data_dict["dt"].append(last_item["timestamp"] - db_rows[-1]["timestamp"])
        except IndexError:
            data_dict["dt"].append(0.01)
        except TypeError:
            data_dict["dt"].append(0.01)

        # Convert lists to torch tensors
        for key in data_dict:
            data_dict[key] = torch.tensor(data_dict[key])
        return data_dict

    def create_aggregated_batched_data_dict(self, durations):
        """
        This method creates a batched set of data from a list of durations. Each duration is individually converted to a batched set of data in order to maintain temporal contiguousness within a batch. These batches are then concatenated.
        """
        batched_data_tuples = []
        for duration in durations:
            data_dict = self.create_data_dict_from_durations(duration)
            st_data_tuple = create_single_track_states_and_controls(data_dict)
            batched_data_tuple = create_batched_track_states_and_controls(
                *st_data_tuple, 5, 10
            )
            batched_data_tuples.append(batched_data_tuple)
        # Concatenate the batched data tuples
        batched_initial_states = torch.cat(
            [data[0] for data in batched_data_tuples], dim=0
        )
        batched_control_inputs = torch.cat(
            [data[1] for data in batched_data_tuples], dim=0
        )
        batched_target_states = torch.cat(
            [data[2] for data in batched_data_tuples], dim=0
        )
        batched_delta_times = torch.cat(
            [data[3] for data in batched_data_tuples], dim=0
        )
        return (
            batched_initial_states,
            batched_control_inputs,
            batched_target_states,
            batched_delta_times,
        )

    def get_timestamp_from_duration_msg(self, duration: Duration):
        """
        This method converts a Duration message to a timestamp.
        """
        # Convert the duration to seconds
        start_seconds = duration.start.sec + duration.start.nanosec * 1e-9
        end_seconds = duration.end.sec + duration.end.nanosec * 1e-9
        return start_seconds, end_seconds

    def fit_parameters(self, cluster_id, durations):
        """
        This method fits the parameters for a given set of durations. It creates a batched set of data from the durations and then uses the SingleTrack model to fit the parameters.
        """
        # Create a batched set of data from the durations
        self.state_db.save_to_disk(os.path.join(self.root_path, "data", "state_db.pth"))
        try:
            (
                batched_initial_states,
                batched_control_inputs,
                batched_target_states,
                batched_delta_times,
            ) = self.create_aggregated_batched_data_dict(durations)
        except ValueError:
            self.get_logger().warn(
                f"Skipping cluster {cluster_id} due to insufficient data."
            )
            return
        # Create a SingleTrack model
        single_track = SingleTrack(**self.context_params[cluster_id]).to(DEVICE)
        param_group = PointParameterGroup(
            self.st_params_list, self.context_params[cluster_id]
        ).to(DEVICE)
        for param in self.st_params_list:
            param_group.disable_gradients(param)
        for param in self.params_list:
            param_group.enable_gradients(param)
        rk4_integrator = RK4(single_track, param_group, timestep=0.01).to(DEVICE)
        try:
            mu_hat_star, (mu_values, mu_losses) = gradient_search_for_mu(
                batched_initial_states=batched_initial_states.to(DEVICE),
                batched_control_inputs=batched_control_inputs.to(DEVICE),
                batched_delta_times=batched_delta_times.to(DEVICE),
                batched_target_states=batched_target_states.to(DEVICE),
                integrator=rk4_integrator,
                num_samples=100,
                epochs=50,
            )
        except RuntimeError:
            self.get_logger().warn(
                f"Skipping cluster {cluster_id} due to runtime error."
            )
            return
        # Update the context parameters with the new mu value
        if isinstance(mu_hat_star, torch.Tensor):
            if torch.isnan(mu_hat_star):
                self.get_logger().warn(
                    f"Skipping cluster {cluster_id} due to NaN mu value."
                )
                return
            mu_hat_value = mu_hat_star.item()
        else:
            if np.isnan(mu_hat_star):
                self.get_logger().warn(
                    f"Skipping cluster {cluster_id} due to NaN mu value."
                )
                return
            mu_hat_value = mu_hat_star
        self.context_params[cluster_id]["mu"] = float(mu_hat_value)

    def publish_complete_estimates(self):
        """
        This method publishes the complete parameter estimates for each cluster_id. It creates a ContextParamMap message and populates it with the parameters for each cluster_id.
        """
        param_map = ContextParamMap()
        param_map.header.stamp = self.get_clock().now().to_msg()
        param_map.header.frame_id = "robot"
        param_map.params = []
        # Populate the ContextParamMap message with the parameters
        for cluster_id in self.context_params.keys():
            context_param_msg = ContextParams()
            context_param_msg.key = cluster_id
            params = self.context_params[cluster_id]
            param_values = []
            for param in self.params_list:
                param_value = ParamValue()
                param_value.name = param
                param_value.value = params[param]
                param_values.append(param_value)
            context_param_msg.params = ParamList(params=param_values)
            param_map.params.append(context_param_msg)
        # Publish the ContextParamMap message
        self.cluster_friction_publisher.publish(param_map)

    def cluster_duration_callback(self, msg):
        """
        This method is called after the clustering process is complete. For each cluster_id, it checks if the durations are different. If the durations have changed, we estimate parameters based on the new durations.
        We then publish the estimated parameters for each cluster_id.
        """
        """
        The message architecture is as follows:
        ContextDurationMap
            header:
            pairs:
                key: cluster-id
                durations:
                    durations:
                        Duration:
                            start: Time
                                sec: int
                                nanosec: int
                            end: Time
                                sec: int
                                nanosec: int
                    

        """
        for pair in msg.pairs:
            cluster_id = pair.key
            self.durations[cluster_id] = []
            for duration in pair.durations.durations:
                # Convert the duration to a tuple (start, end)
                start, end = self.get_timestamp_from_duration_msg(duration)
                self.durations[cluster_id].append((start, end))

        for pair in msg.pairs:
            cluster_id = pair.key
            # Assume the durations have changed and fit new parameters
            if cluster_id not in self.context_params:
                # If the cluster_id is not found, use the default parameters
                self.context_params[cluster_id] = deepcopy(self.default_params)
            self.fit_parameters(cluster_id, self.durations[cluster_id])

        # Publish the complete parameter estimates
        self.publish_complete_estimates()


def main(args=None):
    rclpy.init(args=args)
    node = ParameterEstimator()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt")
    finally:
        executor.shutdown()
        node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
