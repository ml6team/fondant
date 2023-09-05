model = dict(
    type="PAN",
    backbone=dict(type="resnet18", pretrained=True),
    neck=dict(type="FPEM_v1", in_channels=(64, 128, 256, 512), out_channels=128),
    detection_head=dict(
        type="PA_Head",
        in_channels=512,
        hidden_dim=128,
        num_classes=6,
        loss_text=dict(type="DiceLoss", loss_weight=1.0),
        loss_kernel=dict(type="DiceLoss", loss_weight=0.5),
        loss_emb=dict(type="EmbLoss_v1", feature_dim=4, loss_weight=0.25),
    ),
)
data = dict(
    batch_size=16,
    train=dict(
        type="PAN_CTW",
        split="train",
        is_transform=True,
        img_size=640,
        short_size=640,
        kernel_scale=0.7,
        read_type="cv2",
    ),
    test=dict(type="PAN_CTW", split="test", short_size=640, read_type="cv2"),
)
train_cfg = dict(lr=1e-3, schedule="polylr", epoch=600, optimizer="Adam")
test_cfg = dict(
    min_score=0.88, min_area=16, bbox_type="poly", result_path="outputs/submit_ctw/"
)
