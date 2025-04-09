import threading
import numpy as np
import copy
import pickle
import bisect


class ImageDatabase:
    """
    Thread-safe database for storing images with their extracted features and timestamps.
    """

    def __init__(self, max_size=10000):
        """
        Initialize the image database.

        Args:
            max_size: Maximum number of images to store
        """
        self.max_size = max_size
        self.lock = threading.RLock()
        self.items = []  # List of dicts containing image, features, information, timestamp
        self.timestamps = []

    def add(self, features, information, timestamp, **kwargs):
        """
        Add an image with features to the database.

        Args:
            image: The input image
            features: The extracted features dictionary
            information: Combined feature vector
            timestamp: Timestamp of the image

        Returns:
            int: The current database size
        """
        db_entry = {
            "features": features,
            "information": information,
            "timestamp": timestamp,
        }
        db_entry.update(kwargs)

        with self.lock:
            self.items.append(db_entry)

            # Remove oldest entries if database is too large
            while len(self.items) > self.max_size:
                self.items.pop(0)

            return len(self.items)

    def clear(self):
        """
        Clear all items from the database.
        """
        with self.lock:
            self.items = []

    def get_copy(self):
        """
        Get a deep copy of all items in the database.

        Returns:
            list: Copy of all database items
        """
        with self.lock:
            return copy.deepcopy(self.items)

    def get_information_vectors(self):
        """
        Get all feature information vectors stacked as a matrix.

        Returns:
            numpy.ndarray: Matrix with each row being a feature vector
        """
        with self.lock:
            if not self.items:
                return np.array([])
            return np.vstack([item["information"] for item in self.items])

    def get_item(self, index):
        """
        Get an item from the database by index.

        Args:
            index: The index of the item to retrieve

        Returns:
            dict: A copy of the database item at the specified index
            None: If the index is out of range
        """
        with self.lock:
            if 0 <= index < len(self.items):
                return copy.deepcopy(self.items[index])
            return None

    def __getitem__(self, key):
        """
        Enable Python-style indexing and slicing for the database.

        Args:
            key: Integer index or slice object

        Returns:
            dict or list: A copy of the item(s) at the specified index/slice

        Raises:
            IndexError: If the index is out of range
            TypeError: If the key is not an integer or slice
        """
        with self.lock:
            if isinstance(key, (int, slice)):
                return copy.deepcopy(self.items[key])
            else:
                raise TypeError("Database indices must be integers or slices")

    def size(self):
        """
        Get the current size of the database.

        Returns:
            int: Number of items in the database
        """
        with self.lock:
            return len(self.items)

    def save_to_disk(self, filepath):
        """
        Save the database items to disk using pickle serialization.
        Only the items are saved, not the database configuration.

        Args:
            filepath: Path to the file where the database will be saved

        Returns:
            bool: True if successful, False otherwise
        """

        with self.lock:
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(self.items, f)
                return True
            except Exception as e:
                print(f"Error saving database: {e}")
                return False

    @classmethod
    def load_from_disk(cls, filepath, max_size=10000):
        """
        Load a database from disk.
        Creates a new database with the specified max_size and populates it with items from the file.

        Args:
            filepath: Path to the file containing the saved database items
            max_size: Maximum size for the new database

        Returns:
            ImageDatabase: A new database populated with the loaded items
            None: If loading fails
        """

        try:
            with open(filepath, "rb") as f:
                items = pickle.load(f)

            # Create a new database instance
            database = cls(max_size=max_size)

            # Add the loaded items (with thread safety)
            with database.lock:
                # If we have more items than max_size, keep only the most recent ones
                database.items = items[-max_size:] if len(items) > max_size else items

            return database
        except Exception as e:
            print(f"Error loading database: {e}")
            return None


class StateDB:
    """This is a thread-safe database for storing state information."""

    def __init__(self, max_size=10000):
        """
        Initialize the state database.

        Args:
            max_size: Maximum number of states to store
        """
        self.max_size = max_size
        self.lock = threading.RLock()
        self.items = []  # List of dicts containing state information
        self.timestamps = []  # List of timestamps for each state entry

    def add(self, **kwargs):
        """
        Add a state to the database.

        Args:
            kwargs: State information

        Returns:
            int: The current database size
        """
        db_entry = kwargs
        assert "timestamp" in db_entry, "Timestamp is required in the state entry"

        with self.lock:
            self.items.append(db_entry)
            self.timestamps.append(db_entry["timestamp"])

            # Remove oldest entries if database is too large
            while len(self.items) > self.max_size:
                self.items.pop(0)

            return len(self.items)

    def get_copy(self):
        """
        Return a deep copy of all state entries in the database.
        """
        with self.lock:
            return copy.deepcopy(self.items)

    def get_item(self, index):
        """
        Get a state entry by index.

        Args:
            index: The index of the state entry.

        Returns:
            dict: A copy of the state entry if it exists, otherwise None.
        """
        with self.lock:
            if 0 <= index < len(self.items):
                return self.items[index]
            return None

    def __getitem__(self, key):
        """
        Enable Python-style indexing and slicing for the database.

        Args:
            key: An integer index or a slice.

        Returns:
            dict or list: A copy of the state entry (or entries) at the specified index/slice.

        Raises:
            TypeError: If key is not an integer or a slice.
        """
        with self.lock:
            if isinstance(key, (int, slice)):
                return self.items[key]
            raise TypeError("Database indices must be integers or slices")

    def get_index_slice_by_duration(self, start, end):
        """
        Get the closes

        Args:
            start: Start timestamp
            end: End timestamp

        Returns:
            slice: A slice object representing the range of indices that fall within the specified duration.
        """

        # Use binary search on the sorted list of timestamps to find the start and end indices.
        start_index = bisect.bisect_left(self.timestamps, start)
        end_index = bisect.bisect_right(self.timestamps, end) - 1

        if start_index <= end_index:
            return slice(start_index, end_index + 1)
        return slice(None)

    def clear(self):
        """
        Clear all state entries in the database.
        """
        with self.lock:
            self.items = []

    def size(self):
        """
        Get the current number of state entries in the database.

        Returns:
            int: The number of state entries.
        """
        with self.lock:
            return len(self.items)

    def save_to_disk(self, filepath):
        """
        Serialize the state database to disk using pickle.

        Args:
            filepath: Path to the file where the database will be saved.

        Returns:
            bool: True if the save operation is successful, False otherwise.
        """
        with self.lock:
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(self.items, f)
                return True
            except Exception as e:
                print(f"Error saving state database: {e}")
                return False

    @classmethod
    def load_from_disk(cls, filepath, max_size=10000):
        """
        Deserialize the state database from disk.

        Args:
            filepath: Path to the file containing the serialized database.
            max_size: Maximum number of state entries to retain.

        Returns:
            StateDB: A new instance populated with the loaded state entries,
                     or None if the load operation fails.
        """
        try:
            with open(filepath, "rb") as f:
                items = pickle.load(f)
            state_db = cls(max_size=max_size)
            with state_db.lock:
                state_db.items = items[-max_size:] if len(items) > max_size else items
            return state_db
        except Exception as e:
            print(f"Error loading state database: {e}")
            return None


def ns_to_s_and_ns(ns):
    """
    Convert nanoseconds to seconds and nanoseconds.

    Args:
        ns: Time in nanoseconds
    Returns:
        tuple: (seconds, nanoseconds)
    """
    s = ns // 1_000_000_000
    ns = ns % 1_000_000_000
    return int(s), int(ns)


def s_to_s_and_ns(s):
    """
    Convert seconds to seconds and nanoseconds.

    Args:
        s: Time in seconds
    Returns:
        tuple: (seconds, nanoseconds)
    """
    ns = s * 1_000_000_000
    s = ns // 1_000_000_000
    ns = ns % 1_000_000_000
    return int(s), int(ns)
