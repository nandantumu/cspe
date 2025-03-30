"""
This file contains the extraction methods for semantic information the input image
"""

import torch
import clip
from .prompts import CLIP_PROMPTS
from PIL import Image
import cv2


class SemanticExtractor:
    def __init__(self, device="cpu"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=device, jit=False
        )
        self.clip_model.eval()
        self.prompts = CLIP_PROMPTS
        self.processed_prompts = clip.tokenize(self.prompts).to(self.device)

    def process_queue(self, image_queue):
        """
        Processes the image queue
        """
        if len(image_queue) == 0:
            return
        images = torch.stack(image_queue)
        images = self.clip_preprocess(images)
        with torch.no_grad():
            features, _ = self.clip_model(
                images.to(self.device), self.processed_prompts
            )
            features = features.softmax(dim=-1)

        return features.cpu().numpy()

    def __call__(self, img):
        """
        Extracts the semantic information from the image
        """
        if isinstance(img, (list, tuple)):
            img = torch.stack(
                [
                    self.clip_preprocess(
                        Image.fromarray(cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
                    )
                    for i in img
                ]
            )
        else:
            img = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features, _ = self.clip_model(img.to(self.device), self.processed_prompts)
            features = features.softmax(dim=-1)

        return features.cpu().numpy()
