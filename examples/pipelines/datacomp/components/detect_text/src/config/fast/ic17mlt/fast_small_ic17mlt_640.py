model = dict(
    type="FAST",
    backbone=dict(
        type="fast_backbone", config="config/fast/nas-configs/fast_small.config"
    ),
    neck=dict(type="fast_neck", config="config/fast/nas-configs/fast_small.config"),
    detection_head=dict(
        type="fast_head",
        config="config/fast/nas-configs/fast_small.config",
        pooling_size=9,
        loss_text=dict(type="DiceLoss", loss_weight=0.5),
        loss_kernel=dict(type="DiceLoss", loss_weight=1.0),
        loss_emb=dict(type="EmbLoss_v1", feature_dim=4, loss_weight=0.25),
    ),
)
repeat_times = 10
data = dict(
    batch_size=16,
    train=dict(
        type="FAST_IC17MLT",
        split="train",
        is_transform=True,
        img_size=640,
        short_size=640,
        pooling_size=9,
        read_type="cv2",
        repeat_times=repeat_times,
    ),
    test=dict(type="FAST_IC17MLT", split="test", short_size=640, read_type="cv2"),
)
train_cfg = dict(
    lr=1e-3,
    schedule="polylr",
    epoch=300 // repeat_times,
    optimizer="Adam",
    save_interval=10 // repeat_times,
    pretrain="pretrained/fast_small_in1k_epoch_299.pth"
    # https://github.com/czczup/FAST/releases/download/release/fast_small_in1k_epoch_299.pth
)
