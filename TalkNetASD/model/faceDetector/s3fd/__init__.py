import os
import subprocess

import numpy as np
import torch
from torchvision.ops import nms

from .nets import S3FDModel


class S3FD:
    def __init__(self, path, device="cuda"):
        self.device = device
        self.download(path)

        self.net = S3FDModel(device=self.device).to(self.device)
        model_path = os.path.join(os.getcwd(), path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.net.s3fd.load_state_dict(state_dict)
        self.net.eval()

    def download(self, path):
        if os.path.isfile(path) == False:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
            cmd = "gdown --id %s -O %s" % (Link, path)
            subprocess.call(cmd, shell=True, stdout=None)

        self.img_mean = torch.tensor([123.0, 117.0, 104.0])
        self.img_mean = self.img_mean[:, np.newaxis, np.newaxis]
        self.img_mean = self.img_mean.float().cuda()

    def compile(self, inputs):
        self.net.compile(inputs)

    @torch.no_grad()
    def detect_faces(self, images, scale, conf_th=0.8):
        # images must be already scaled down
        # scale contains [w, h, w, h] of the original image size
        b, c, h, w = images.shape

        batch_bboxes = []

        images -= self.img_mean
        detections = self.net(images).to(self.device)

        scores_mask = detections[:, :, :, 0] > conf_th
        pts = detections[:, :, :, 1:] * scale

        for b_idx in range(b):
            bboxes = torch.hstack((
                pts[b_idx, scores_mask[b_idx]],
                detections[b_idx, :, :, 0][scores_mask[b_idx]].reshape(-1, 1))
            )
            keep = nms(bboxes[:, :4], bboxes[:, 4], 0.1)
            batch_bboxes.append(bboxes[keep])

        return batch_bboxes
