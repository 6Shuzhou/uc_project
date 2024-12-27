import os

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=["geospatial_fm"])
num_frames = 6
img_size = 366

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "Current_Pretrained_Prithvi_Weights\Prithvi_EO_V1_100M.pt" # Make Folder named Current_Pretrained_Prithvi_Weights and Move Model Weights to Folder in hls-foundation-os
num_layers = 6
patch_size = 61
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 10
eval_epoch_interval = 1

loss_weights_multi = [
    0.18119003582799867,
    1.179218931128637,
    8.914099263452227,
    0.615736233042129,
    41.809536729488975,
    2.1913061063255275,
    20.608246827655915,
    0.8842357570937514,
    1.44213778969377,
    3.060547661851139,
    4.612657619208314,
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim * num_frames


# TO BE DEFINED BY USER: Save directory
experiment = "Size_10_Experiment_3_Fold_1_Setting_1"
project_dir = "UC_Project"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "Current_Project_Experiment_Data" # Make Folder named Current_Project_Experiment_Data and Move Necessary Data to Folder in hls-foundation-os

img_norm_cfg = dict(
    means=[
        1055.40380859375,
        1007.68310546875,
        956.6643676757812,
        971.6793823242188,
        951.88330078125,
        1024.5235595703125,
        1072.9144287109375,
        1135.9691162109375,
        1049.761474609375,
        1019.1922607421875,
        984.0466918945312,
        907.1197509765625,
        956.561767578125,
        926.6974487304688,
        824.222412109375,
        795.5777587890625,
        725.1712646484375,
        877.7467041015625,
        1060.6304931640625,
        1114.178466796875,
        1051.0791015625,
        916.0443115234375,
        815.7448120117188,
        682.3311157226562,
    ],
    stds=[
        313.54681396484375,
        195.52859497070312,
        79.60260009765625,
        108.04751586914062,
        208.7662811279297,
        75.85675811767578,
        60.99747085571289,
        197.2439727783203,
        64.9381332397461,
        312.95953369140625,
        318.49774169921875,
        406.10906982421875,
        377.6221923828125,
        216.6307373046875,
        123.1187973022461,
        149.892822265625,
        251.6231231689453,
        84.84525299072266,
        119.59475708007812,
        206.69406127929688,
        106.04676818847656,
        356.41400146484375,
        374.7099304199219,
        483.1792907714844,
    ],
)

bands = [0, 1, 2, 3]

tile_size = 366
crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="LoadGeospatialAnnotations", reduce_zero_label=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="ToTensor", keys=["img", "gt_semantic_seg"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(type="TorchRandomCrop", crop_size=crop_size),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, tile_size, tile_size),
    ),
    dict(type="Reshape", keys=["gt_semantic_seg"], new_shape=(1, tile_size, tile_size)),
    dict(type="CastTensor", keys=["gt_semantic_seg"], new_type="torch.LongTensor"),
    dict(type="Collect", keys=["img", "gt_semantic_seg"]),
]

test_pipeline = [
    dict(type="LoadGeospatialImageFromFile", to_float32=True),
    dict(type="ToTensor", keys=["img"]),
    # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type="TorchNormalize", **img_norm_cfg),
    dict(
        type="Reshape",
        keys=["img"],
        new_shape=(len(bands), num_frames, -1, -1),
        look_up=dict({"2": 1, "3": 2}),
    ),
    dict(type="CastTensor", keys=["img"], new_type="torch.FloatTensor"),
    dict(
        type="CollectTestList",
        keys=["img"],
        meta_keys=[
            "img_info",
            "seg_fields",
            "img_prefix",
            "seg_prefix",
            "filename",
            "ori_filename",
            "img",
            "img_shape",
            "ori_shape",
            "pad_shape",
            "scale_factor",
            "img_norm_cfg",
        ],
    ),
]

CLASSES = (
    "Wheat",
    "Maize",
    "Sorghum",
    "Barley",
    "Rye",
    "Oats",
    "Grapes",
    "Rapeseed",
    "Sunflower",
    "Potatoes",
    "Peas",
)

dataset = "GeospatialDataset"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="Training_Set",
        ann_dir="Training_Set",
        pipeline=train_pipeline,
        img_suffix="_image.npy",
        seg_map_suffix="_labels.npy",
    ),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="Validation_Set",
        ann_dir="Validation_Set",
        pipeline=test_pipeline,
        img_suffix="_image.npy",
        seg_map_suffix="_labels.npy",
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir="Test_Set",
        ann_dir="Test_Set",
        pipeline=test_pipeline,
        img_suffix="_image.npy",
        seg_map_suffix="_labels.npy",
    ),
)

optimizer = dict(type="Adam", lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)
log_config = dict(
    interval=10, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

checkpoint_config = dict(by_epoch=True, interval=100, out_dir=save_path)

evaluation = dict(
    interval=eval_epoch_interval,
    metric=['mIoU', 'mDice', 'mFscore'], # ['mIoU', 'mDice', 'mFscore'], # ["aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall"], # "mIoU",
    pre_eval=True,
    save_best="mIoU",
    by_epoch=True,
)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)
workflow = [("train", 1)]
norm_cfg = dict(type="BN", requires_grad=True)

model = dict(
    type="TemporalEncoderDecoder",
    frozen_backbone=False,
    backbone=dict(
        type="TemporalViTEncoder",
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ),
    neck=dict(
        type="ConvTransformerTokensToEmbeddingNeck",
        embed_dim=embed_dim * num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=img_size // patch_size,
        Wp=img_size // patch_size,
    ),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type="FCNHead",
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=dict(type="BN", requires_grad=True),
        align_corners=False,
        loss_decode=loss_func,
    ),
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        stride=(int(tile_size / 2), int(tile_size / 2)),
        crop_size=(tile_size, tile_size),
    ),
)
auto_resume = False
