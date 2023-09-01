import torch
import torch.nn as nn
from models.loss import build_loss, ohem_batch, iou
from models.utils.nas_utils import set_layer_from_config
from models.utils import generate_bbox
import torch.nn.functional as F
import numpy as np
import json
import cv2

# try:
import pyximport

pyximport.install()

ccl_cuda_success = False
try:
    # from ..post_processing import ccl_cuda
    from models.post_processing.ccl import ccl_cuda

    ccl_cuda_success = True
    print("ccl_cuda successfully not installed!")
except:
    print("ccl_cuda is not installed!")


class FASTHead(nn.Module):
    def __init__(
        self,
        conv,
        blocks,
        final,
        pooling_size,
        loss_text,
        loss_kernel,
        loss_emb,
        dropout_ratio=0,
    ):
        super(FASTHead, self).__init__()
        self.conv = conv
        if blocks is not None:
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = None
        self.final = final

        self.text_loss = build_loss(loss_text)
        self.kernel_loss = build_loss(loss_kernel)
        self.emb_loss = build_loss(loss_emb)

        self.pooling_size = pooling_size

        self.pooling_1s = nn.MaxPool2d(
            kernel_size=self.pooling_size,
            stride=1,
            padding=(self.pooling_size - 1) // 2,
        )
        self.pooling_2s = nn.MaxPool2d(
            kernel_size=self.pooling_size // 2 + 1,
            stride=1,
            padding=(self.pooling_size // 2) // 2,
        )

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        if self.blocks is not None:
            for block in self.blocks:
                x = block(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.final(x)
        return x

    def get_results(self, out, img_meta, cfg, scale=2):
        # if not self.training:
        #     torch.cuda.synchronize()
        #     start = time.time()

        org_img_size = img_meta["org_img_size"][0]
        img_size = img_meta["img_size"][0]  # 640*640
        batch_size = out.size(0)
        outputs = dict()

        texts = F.interpolate(
            out[:, 0:1, :, :],
            size=(img_size[0] // scale, img_size[1] // scale),
            mode="nearest",
        )  # B*1*320*320
        texts = self._max_pooling(texts, scale=scale)  # B*1*320*320
        score_maps = torch.sigmoid_(texts)  # B*1*320*320
        score_maps = F.interpolate(
            score_maps, size=(img_size[0], img_size[1]), mode="nearest"
        )  # B*1*640*640
        score_maps = score_maps.squeeze(1)  # B*640*640

        kernels = (out[:, 0, :, :] > 0).to(torch.uint8)  # B*160*160
        if kernels.is_cuda and ccl_cuda_success:
            labels_ = ccl_cuda.ccl_batch(kernels)  # B*160*160
        else:
            labels_ = []
            for kernel in kernels.cpu().numpy():
                ret, label_ = cv2.connectedComponents(kernel)
                labels_.append(label_)
            labels_ = np.array(labels_)
            labels_ = torch.from_numpy(labels_)
        labels = labels_.unsqueeze(1).to(torch.float32)  # B*1*160*160
        labels = F.interpolate(
            labels, size=(img_size[0] // scale, img_size[1] // scale), mode="nearest"
        )  # B*1*320*320
        labels = self._max_pooling(labels, scale=scale)
        labels = F.interpolate(
            labels, size=(img_size[0], img_size[1]), mode="nearest"
        )  # B*1*640*640
        labels = labels.squeeze(1).to(torch.int32)  # B*640*640

        keys = [torch.unique(labels_[i], sorted=True) for i in range(batch_size)]

        # if not self.training:
        #     torch.cuda.synchronize()
        #     outputs.update(dict(post_time=time.time() - start))

        outputs.update(dict(kernels=kernels.data.cpu()))

        scales = (
            float(org_img_size[1]) / float(img_size[1]),
            float(org_img_size[0]) / float(img_size[0]),
        )

        results = []
        for i in range(batch_size):
            bboxes, scores = generate_bbox(
                keys[i], labels[i], score_maps[i], scales, cfg
            )
            results.append(dict(bboxes=bboxes, scores=scores))
        outputs.update(dict(results=results))

        return outputs

    def _max_pooling(self, x, scale=1):
        if scale == 1:
            x = self.pooling_1s(x)
        elif scale == 2:
            x = self.pooling_2s(x)
        return x

    def loss(self, out, gt_texts, gt_kernels, training_masks, gt_instances):
        # output
        kernels = out[:, 0, :, :]  # 4*640*640
        texts = self._max_pooling(kernels, scale=1)  # 4*640*640
        embs = out[:, 1:, :, :]  # 4*4*640*640

        # text loss
        selected_masks = ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.text_loss(texts, gt_texts, selected_masks, reduce=False)
        iou_text = iou((texts > 0).long(), gt_texts, training_masks, reduce=False)
        losses = dict(loss_text=loss_text, iou_text=iou_text)

        # kernel loss
        selected_masks = gt_texts * training_masks
        loss_kernel = self.kernel_loss(
            kernels, gt_kernels, selected_masks, reduce=False
        )
        loss_kernel = torch.mean(loss_kernel, dim=0)
        iou_kernel = iou((kernels > 0).long(), gt_kernels, selected_masks, reduce=False)
        losses.update(dict(loss_kernels=loss_kernel, iou_kernel=iou_kernel))

        # auxiliary loss
        loss_emb = self.emb_loss(
            embs, gt_instances, gt_kernels, training_masks, reduce=False
        )
        losses.update(dict(loss_emb=loss_emb))

        return losses

    @staticmethod
    def build_from_config(config, **kwargs):
        conv = set_layer_from_config(config["conv"])
        final = set_layer_from_config(config["final"])
        try:
            blocks = []
            for block_config in config["blocks"]:
                blocks.append(set_layer_from_config(block_config))
            return FASTHead(conv, blocks, final, **kwargs)
        except:
            return FASTHead(conv, None, final, **kwargs)


def fast_head(config, **kwargs):
    head_config = json.load(open(config, "r"))["head"]
    head = FASTHead.build_from_config(head_config, **kwargs)
    return head
