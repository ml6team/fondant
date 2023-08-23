import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head


class FAST(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(FAST, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode="bilinear")

    def forward(
        self,
        imgs,
        gt_texts=None,
        gt_kernels=None,
        training_masks=None,
        gt_instances=None,
        img_metas=None,
        cfg=None,
    ):
        outputs = dict()

        if not self.training:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # reduce channel
        f = self.neck(f)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training:
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size(), scale=1)
            det_loss = self.det_head.loss(
                det_out, gt_texts, gt_kernels, training_masks, gt_instances
            )
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), scale=4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg, scale=2)
            outputs.update(det_res)

        return outputs
