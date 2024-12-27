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
    0.18268539228902722,
    1.2117552013830226,
    9.586243489764701,
    0.6043839067418172,
    28.5807133268304,
    2.352732488066731,
    25.200953083299414,
    0.8556729653191001,
    1.6351758773978133,
    2.426888109995621,
    4.000283327169345,
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim * num_frames


# TO BE DEFINED BY USER: Save directory
experiment = "Size_10_Experiment_3_Fold_3_Setting_2"
project_dir = "UC_Project"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "Current_Project_Experiment_Data" # Make Folder named Current_Project_Experiment_Data and Move Necessary Data to Folder in hls-foundation-os

img_norm_cfg = dict(
    means=[
        1066.5335693359375,
        1122.56591796875,
        1046.3907470703125,
        1043.1103515625,
        972.1835327148438,
        892.26025390625,
        958.8492431640625,
        941.6051635742188,
        835.4815063476562,
        803.6522827148438,
        723.397216796875,
        856.4780883789062,
        1053.588134765625,
        1098.3687744140625,
        1042.8162841796875,
        934.7946166992188,
        796.62841796875,
        663.6910400390625,
        2149.4443359375,
        2496.18359375,
        3000.01806640625,
        3381.552978515625,
        3851.508544921875,
        3737.052001953125,
        3594.255615234375,
        3227.5107421875,
        2956.452880859375,
        2742.67822265625,
        2502.8564453125,
        2236.89306640625,
        1687.8787841796875,
        1996.8912353515625,
        2055.3173828125,
        1989.720703125,
        1846.309326171875,
        2050.0595703125,
    ],
    stds=[
        56.582218170166016,
        201.6793975830078,
        59.95194625854492,
        386.39349365234375,
        295.9848327636719,
        405.9190673828125,
        346.82025146484375,
        220.212890625,
        125.02291870117188,
        155.28089904785156,
        265.9369201660156,
        91.34030151367188,
        117.70952606201172,
        214.47079467773438,
        101.5183334350586,
        435.59033203125,
        346.1260986328125,
        480.68194580078125,
        318.8410339355469,
        173.874755859375,
        139.62013244628906,
        138.2492218017578,
        198.69776916503906,
        133.00527954101562,
        117.23890686035156,
        189.41029357910156,
        146.4978485107422,
        431.9341125488281,
        350.38262939453125,
        550.0335083007812,
        381.45599365234375,
        149.4227752685547,
        115.72653198242188,
        146.7903594970703,
        272.1116027832031,
        96.24225616455078,
    ]
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
