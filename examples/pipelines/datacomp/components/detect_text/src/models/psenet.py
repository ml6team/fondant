import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head


class PSENet(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode="bilinear")

    def forward(
        self,
        imgs,
        gt_texts=None,
        gt_kernels=None,
        training_masks=None,
        img_metas=None,
        cfg=None,
    ):
        outputs = dict()

        if not self.training and hasattr(cfg, "report_speed"):
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)
        if not self.training and hasattr(cfg, "report_speed"):
            torch.cuda.synchronize()
            outputs.update(dict(backbone_time=time.time() - start))
            start = time.time()

        # FPN
        (
            f1,
            f2,
            f3,
            f4,
        ) = self.fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)

        if not self.training and hasattr(cfg, "report_speed"):
            torch.cuda.synchronize()
            outputs.update(dict(neck_time=time.time() - start))
            start = time.time()

        # detection

        det_out = self.det_head(f)

        if not self.training and hasattr(cfg, "report_speed"):
            torch.cuda.synchronize()
            outputs.update(dict(det_head_time=time.time() - start))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs
