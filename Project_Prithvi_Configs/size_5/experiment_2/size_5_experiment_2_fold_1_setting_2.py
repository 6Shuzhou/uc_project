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
    97.09133094283906,
    1.2254172042743583,
    18.17300100015792,
    0.15336231150597152,
    11.860653872085683,
    1.075880283747619,
    0.6103063022291894,
    2.1777211460976442,
    6.302796009091822,
    55.000697002887584,
    3.228138828694256,
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim * num_frames


# TO BE DEFINED BY USER: Save directory
experiment = "Size_5_Experiment_2_Fold_1_Setting_2"
project_dir = "UC_Project"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "Current_Project_Experiment_Data" # Make Folder named Current_Project_Experiment_Data and Move Necessary Data to Folder in hls-foundation-os

img_norm_cfg = dict(
    means=[
        1143.1483154296875,
        1126.531494140625,
        1076.4971923828125,
        1055.6929931640625,
        1035.22021484375,
        1152.88525390625,
        860.0780029296875,
        903.6128540039062,
        921.4554443359375,
        951.9763793945312,
        975.3358764648438,
        1100.1956787109375,
        1155.04541015625,
        1153.7811279296875,
        1064.8714599609375,
        1031.0091552734375,
        972.9629516601562,
        1075.439208984375,
        2042.9462890625,
        2246.465087890625,
        2475.72021484375,
        2760.988037109375,
        2997.870849609375,
        2935.36572265625,
        2928.08740234375,
        2846.848876953125,
        2599.95263671875,
        2371.067626953125,
        2192.933837890625,
        2217.760009765625,
        1577.840087890625,
        1782.511474609375,
        1906.4468994140625,
        1995.9678955078125,
        2032.442138671875,
        2260.21435546875,
    ],
    stds=[
        86.97515869140625,
        81.87934875488281,
        187.09947204589844,
        215.2753448486328,
        277.0019226074219,
        457.9858093261719,
        143.9781036376953,
        112.9113998413086,
        118.0002670288086,
        138.5314483642578,
        179.52352905273438,
        138.0922393798828,
        129.71875,
        127.41870880126953,
        204.29367065429688,
        234.64990234375,
        305.0632629394531,
        505.2449645996094,
        242.3201141357422,
        185.57101440429688,
        186.90908813476562,
        162.7401885986328,
        168.4741668701172,
        161.7404327392578,
        147.41851806640625,
        156.9661102294922,
        229.9902801513672,
        263.9158630371094,
        315.09857177734375,
        496.5655212402344,
        179.65341186523438,
        156.92869567871094,
        138.79598999023438,
        145.9631805419922,
        175.92144775390625,
        153.322509765625,
    ],
)

bands = [0, 1, 2, 3, 4, 5]

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