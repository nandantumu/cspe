"""
This module contains a class that extracts semantic and low level information from an image. It does so by either using batched processing or single image processing. The class maintains a database of images, their extracted information, and the timestamp of the provided image. Using this database of images, the class runs DBSCAN clustering to on the extracted information to identify the context of the image.
"""

import threading
import datetime
import numpy as np
from sklearn.cluster import HDBSCAN
from scipy.spatial import KDTree
import copy
from typing import List

from .low_level_info import LowLevelExtractor
from .semantic_info import SemanticExtractor
from ..utilities import ImageDatabase


class ContextIdentifier:
    """
    A class that extracts semantic and low-level information from images and
    identifies the context by clustering the extracted features.
    Thread-safe database of images maintained in memory.
    """

    def __init__(
        self, low_level_extractor=None, semantic_extractor=None, max_database_size=10000
    ):
        """
        Initialize the context identifier.

        Args:
            low_level_extractor: The low level feature extractor. If None, a default one is initialized.
            semantic_extractor: The semantic feature extractor. If None, a default one is initialized.
            max_database_size: Maximum number of images to store in the database.
        """
        self.low_level_extractor = (
            low_level_extractor if low_level_extractor else LowLevelExtractor()
        )
        self.semantic_extractor = (
            semantic_extractor if semantic_extractor else SemanticExtractor()
        )

        # Replace direct database with ImageDatabase instance
        self.image_db = ImageDatabase(max_size=max_database_size)

        # DBSCAN parameters
        self.dbscan = None
        self.eps = 0.5  # Default DBSCAN distance threshold
        self.min_samples = 100  # Default minimum samples for DBSCAN

        # Thread-safe batch queue
        self.queue_lock = threading.RLock()
        self.batch_queue = []  # List of tuples (image, timestamp)

        self.clusters = []

    def process_single_image(self, image, timestamp=None):
        """
        Process a single image and update the database.

        Args:
            image: The input image
            timestamp: Optional timestamp. If None, current time is used.

        """
        # Extract timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now()

        # Extract features
        low_level_features = self.low_level_extractor.extract(image)
        semantic_features = self.semantic_extractor.extract(image)

        # Combine features
        separate_information = {
            "low_level": low_level_features,
            "semantic": semantic_features,
        }
        information = np.concatenate([low_level_features, semantic_features], axis=0)

        # Update database using the ImageDatabase object
        self.image_db.add(separate_information, information, timestamp)

    def add_to_queue(self, image, timestamp=None):
        """
        Add an image to the batch processing queue.

        Args:
            image: The input image
            timestamp: Optional timestamp. If None, current time is used.

        Returns:
            int: Current size of the queue
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()

        with self.queue_lock:
            self.batch_queue.append((image, timestamp))
            return len(self.batch_queue)

    def process_queue(self):
        """
        Process all images in the batch queue.

        Returns:
            list: List of dictionaries with extracted features and context information
        """
        with self.queue_lock:
            if not self.batch_queue:
                return []

            # Make a local copy of the queue and clear it
            queue_copy = self.batch_queue.copy()
            self.batch_queue = []

        # Process the batched images
        images, timestamps = zip(*queue_copy)
        return self.process_batch(images, timestamps)

    def process_batch(self, images, timestamps=None):
        """
        Process a batch of images.

        Args:
            images: List of input imagdfes
            timestamps: Optional list of timestamps. If None, current times are used.

        Returns:
            list: List of dictionaries with extracted features and context information
        """
        if timestamps is None:
            timestamps = [datetime.datetime.now() for _ in images]
        elif len(timestamps) != len(images):
            raise ValueError("Number of timestamps must match number of images")

        # Batch process image context to improve performance
        low_level_features = self.low_level_extractor(images)
        semantic_features = self.semantic_extractor(images)
        # Combine features
        information_vectors = np.concatenate(
            [low_level_features, semantic_features], axis=1
        )
        cluster_estimates = self.find_nearest_context(
            [information_vectors[i] for i in range(information_vectors.shape[0])]
        )

        for i in range(len(images)):
            separate_information = {
                "low_level": low_level_features[i],
                "semantic": semantic_features[i],
            }

            # Update database using ImageDatabase object
            self.image_db.add(
                separate_information, information_vectors[i], timestamps[i]
            )

        return cluster_estimates

    def find_nearest_context(self, features: List[np.ndarray]) -> List[int]:
        """
        Find the nearest context for a given set of features.

        Args:
            features: List of feature vectors

        Returns:
            list: List of nearest cluster IDs
        """
        if self.image_db.size() == 0:
            return []

        try:
            # Find nearest neighbors
            distances, indices = self.tree.query(features, k=1)

            # Get the corresponding cluster IDs
            nearest_clusters = [self.clusters[i] for i in indices]
        except AttributeError:
            nearest_clusters = [-1] * len(features)

        return nearest_clusters

    def reprocess_clusters(self):
        """
        Identify context using DBSCAN clustering.

        Args:
            features: Features of the current image

        Returns:
            dict: Context information including cluster ID and similarity
        """
        # If database is too small, return default context
        if self.image_db.size() < self.min_samples:
            return -1

        # Extract feature vectors from all images in database
        all_features = self.image_db.get_information_vectors()

        # Include current feature vector for classification
        all_features = np.vstack([all_features])

        self.tree = KDTree(all_features)

        # Fit DBSCAN
        self.dbscan = HDBSCAN(
            min_cluster_size=self.min_samples, n_jobs=4, allow_single_cluster=True
        )
        labels = self.dbscan.fit_predict(all_features)

        persistent_labels = self.ensure_cluster_persistence(labels)

        self.clusters = persistent_labels

        # The last label corresponds to the current image
        current_label = persistent_labels[-1]

        return int(current_label)

    def ensure_cluster_persistence(self, new_labels):
        """
        Ensure that the clusters are persistent across different runs of DBSCAN.

        Args:
            new_labels: New cluster labels from DBSCAN

        Returns:
            list: Updated cluster labels
        """
        # If there are no previous clusters, just return the new labels
        if len(self.clusters) == 0:
            return new_labels

        # Calculate centroids for previous clusters
        prev_unique_labels = np.unique(self.clusters)
        prev_centroids = {}

        for prev_label in prev_unique_labels:
            if prev_label == -1:  # Skip noise points
                continue

            # Get indices of points in this cluster
            indices = np.where(self.clusters == prev_label)[0]

            # Calculate centroid using feature vectors
            features = np.vstack(
                [self.image_db[int(idx)]["information"] for idx in indices]
            )
            centroid = np.mean(features, axis=0)
            prev_centroids[prev_label] = centroid

        # Calculate centroids for new clusters
        new_unique_labels = np.unique(new_labels)
        new_centroids = {}

        for new_label in new_unique_labels:
            if new_label == -1:  # Skip noise points
                continue

            # Get indices of points in this cluster
            indices = np.where(new_labels == new_label)[0]

            # Calculate centroid
            features = np.vstack(
                [self.image_db[int(idx)]["information"] for idx in indices]
            )
            centroid = np.mean(features, axis=0)
            new_centroids[new_label] = centroid

        # Create a mapping from new labels to old labels
        label_mapping = {-1: -1}  # Noise points stay as noise

        # Distance threshold for considering centroids as the same cluster
        threshold = 1e6

        # For each previous centroid, find the closest new centroid
        for prev_label, prev_centroid in prev_centroids.items():
            closest_new_label = None
            min_distance = float("inf")

            for new_label, new_centroid in new_centroids.items():
                # Skip if this new label is already mapped
                if new_label in label_mapping:
                    continue

                # Calculate Euclidean distance between centroids
                distance = np.linalg.norm(prev_centroid - new_centroid)

                if distance < min_distance:
                    min_distance = distance
                    closest_new_label = new_label

            # If the closest centroid is within threshold, assign the previous ID
            if closest_new_label is not None and min_distance < threshold:
                label_mapping[closest_new_label] = prev_label

        # Assign new IDs to unmapped clusters
        max_label = max(
            [label for label in prev_unique_labels if label != -1], default=0
        )
        for new_label in new_unique_labels:
            if new_label == -1:
                continue
            if new_label not in label_mapping:
                max_label += 1
                label_mapping[new_label] = max_label

        # Apply the mapping to create updated labels
        updated_labels = np.array(
            [label_mapping.get(label, label) for label in new_labels]
        )

        return updated_labels

    def get_cluster_durations(self):
        """
        Get the durations of each cluster.

        Returns:
            list: List of tuples with cluster ID and duration
        """
        cluster_durations = {key: [] for key in np.unique(self.clusters)}
        # Deposit start and end indexes for each contiguous block of the same cluster.
        if len(self.clusters) == 0:
            return cluster_durations

        current_cluster = self.clusters[0]
        start_index = 0
        for i in range(1, len(self.clusters)):
            if self.clusters[i] != current_cluster:
                cluster_durations[current_cluster].append((start_index, i - 1))
                current_cluster = self.clusters[i]
                start_index = i
        cluster_durations[current_cluster].append((start_index, len(self.clusters) - 1))

        cluster_duration_timestamps = {key: [] for key in cluster_durations.keys()}
        # Calculate the duration for each cluster by looking up the timestamps in the database
        for cluster_id, intervals in cluster_durations.items():
            for start, end in intervals:
                start_time = self.image_db[start]["timestamp"]
                end_time = self.image_db[end]["timestamp"]
                cluster_duration_timestamps[cluster_id].append((start_time, end_time))

        return cluster_duration_timestamps

    def get_database(self):
        """
        Get a copy of the current database.

        Returns:
            list: Copy of the image database
        """
        return self.image_db.get_copy()

    def clear_database(self):
        """
        Clear the image database.
        """
        self.image_db.clear()

    def set_clustering_parameters(self, eps=None, min_samples=None):
        """
        Update DBSCAN clustering parameters.

        Args:
            eps: DBSCAN eps parameter (distance threshold)
            min_samples: DBSCAN min_samples parameter
        """
        if eps is not None:
            self.eps = eps
        if min_samples is not None:
            self.min_samples = min_samples
