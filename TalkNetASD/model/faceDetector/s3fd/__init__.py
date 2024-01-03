import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torch_tensorrt

from .box_utils import nms_
from .nets import S3FDNet


class S3FD:
    def __init__(self, path, device="cuda"):
        self.device = device
        self.download(path)

        self.net = S3FDNet(device=self.device).to(self.device)
        model_path = os.path.join(os.getcwd(), path)
        state_dict = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.compiled = False

    def download(self, path):
        if os.path.isfile(path) == False:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
            cmd = "gdown --id %s -O %s" % (Link, path)
            subprocess.call(cmd, shell=True, stdout=None)

        self.img_mean = torch.tensor([123.0, 117.0, 104.0])
        self.img_mean = self.img_mean[:, np.newaxis, np.newaxis]
        self.img_mean = self.img_mean.float().cuda()

    @torch.no_grad()
    def detect_faces(self, images, conf_th=0.8, scales=[1]):
        images = torch.permute(images, (0, 3, 1, 2))
        b, c, h, w = images.shape

        batch_bboxes = []

        for s in scales:
            scaled_img = F.interpolate(images.float(), size=(int(s * h), int(s * w)))
            scaled_img -= self.img_mean

            if not self.compiled:
                self.net = torch_tensorrt.compile(
                    self.net,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=[1, c, h, w],
                            opt_shape=[b, c, h, w],
                            max_shape=[b, c, h, w],
                            dtype=torch.float16,
                        )
                    ],
                    enabled_precisions={torch_tensorrt.dtype.half},  # Run with FP16
                )
                self.compiled = True

            detections = self.net(scaled_img)
            scale = torch.Tensor([w, h, w, h])

            scores_mask = detections[:, :, :, 0] > conf_th
            pts = detections[:, :, :, 1:] * scale

            for b_idx in range(b):
                bboxes = torch.hstack((
                    pts[b_idx, scores_mask[b_idx]],
                    detections[b_idx, :, :, 0][scores_mask[b_idx]].reshape(-1, 1))
                ).cpu().numpy()
                keep = nms_(bboxes, 0.1)
                batch_bboxes.append(bboxes[keep])

        return batch_bboxes
