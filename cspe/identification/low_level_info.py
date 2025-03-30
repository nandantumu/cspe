"""
This file contains the extraction methods for low level information from the input image
"""

import numpy as np
from .prompts import CLIP_PROMPTS


class LowLevelExtractor:
    def __init__(self):
        self.image_queue = []

    def add_img(self, img):
        """
        Extracts the semantic information from the image.

        """
        self.image_queue.append(img)

    def process_queue(self):
        """
        Processes the image queue
        """
        if len(self.image_queue) == 0:
            return
        elif len(self.image_queue) == 1:
            avg_rgb = self(self.image_queue[0])
        else:
            images = np.stack(self.image_queue)
            avg_rgb = images.mean(axis=(1, 2))

        self.image_queue = []
        return avg_rgb

    def __call__(self, img):
        """
        Extracts the semantic information from the image.
        Image may be a single image, with or without a batch dimension, or a list or tuple of images.
        """
        if isinstance(img, (list, tuple)):
            img = np.stack(img)
        if len(img.shape) == 3:
            avg_rgb = img.mean(axis=(0, 1))
        elif len(img.shape) == 4:
            avg_rgb = img.mean(axis=(1, 2))
        else:
            raise ValueError("Invalid image shape")

        return avg_rgb
