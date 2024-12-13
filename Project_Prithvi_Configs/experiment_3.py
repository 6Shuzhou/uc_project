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
pretrained_weights_path = r"C:\Users\Gebruiker\Documents\LU%20UC\Project\uc_project\Pretrained_Prithvi_Weights\Prithvi_EO_V1_100M.pt" # Absolute Path to Data Root Directory, replace Spaces with %20
num_layers = 6
patch_size = 61
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 10
eval_epoch_interval = 1

loss_weights_multi = [
    #0,
    0.16372301731024866,
    6.2666786550581675,
    0,
    2.7315591487605886,
    0.9884239340771169,
    103.95713879849166,
    2.8451148998734723,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    12.940248387698283,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    2.0773271669164615,
    0.29809647660686833,
    0,
    0,
    371.6539704524469,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    12.874684131401336,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0
]
loss_func = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True,
)
output_embed_dim = embed_dim * num_frames


# TO BE DEFINED BY USER: Save directory
experiment = "Experiment_3"
project_dir = "UC_Project"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = r"C:\Users\Gebruiker\Documents\LU%20UC\Project\uc_project\Experiments_Transformed_Selected_Subset\Experiment_3"  # Absolute Path to Data Root Directory, replace Spaces with %20

img_norm_cfg = dict(
    means=[
        938.8909912109375,
        1165.7423095703125,
        969.1386108398438,
        998.5596923828125,
        1001.1380615234375,
        1053.4913330078125,
        1081.6075439453125,
        1057.3912353515625,
        1047.5184326171875,
        1082.989013671875,
        1125.8248291015625,
        1166.029541015625,
        814.0618286132812,
        1099.354736328125,
        879.5530395507812,
        870.7994995117188,
        808.5398559570312,
        918.6882934570312,
        1037.742431640625,
        976.7101440429688,
        1007.8046875,
        1046.3336181640625,
        1094.3031005859375,
        1140.5767822265625,
    ],
    stds=[
        45.462974548339844,
        360.610595703125,
        125.10393524169922,
        123.52023315429688,
        141.56320190429688,
        70.17779541015625,
        72.74740600585938,
        96.46942901611328,
        79.5517807006836,
        156.83705139160156,
        335.3448791503906,
        513.7928466796875,
        62.85222625732422,
        378.9572448730469,
        174.2990264892578,
        192.8314666748047,
        227.90185546875,
        107.56732177734375,
        156.86184692382812,
        141.8263702392578,
        122.671142578125,
        187.54225158691406,
        369.41839599609375,
        561.60546875,
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
    #"Background/Other",
    "Wheat",
    "Maize",
    "Rice",
    "Sorghum",
    "Barley",
    "Rye",
    "Oats",
    "Millets",
    "Mixed cereals",
    "Other",
    "Leafy or stem vegetables",
    "Artichokes",
    "Cauliflowers and broccoli",
    "Lettuce",
    "Spinach",
    "Other leafy or stem vegetables n.e.c.",
    "Cucumbers",
    "Eggplants (aubergines)",
    "Tomatoes",
    "Watermelons",
    "Cantaloupes and other melons",
    "Pumpkin squash and gourds",
    "Root bulb or tuberous vegetables",
    "Carrots",
    "Turnips",
    "Onions (incl. shallots)",
    "Leeks and other alliaceous vegetables",
    "Other root bulb or tuberous vegetables n.e.c.",
    "Mushrooms and truffles",
    "Figs",
    "Other tropical and subtropical fruits n.e.c.",
    "Grapefruit and pomelo",
    "Lemons and Limes",
    "Oranges",
    "Tangerines mandarins clementines",
    "Grapes",
    "Kiwi fruit",
    "Apples",
    "Apricots",
    "Cherries and sour cherries",
    "Peaches and nectarines",
    "Pears and quinces",
    "Plums and sloes",
    "Other pome fruits and stone fruits n.e.c.",
    "Almonds",
    "Chestnuts",
    "Hazelnuts",
    "Pistachios",
    "Walnuts",
    "Other nuts n.e.c.",
    "Other fruits",
    "Soya beans",
    "Rapeseed",
    "Sunflower",
    "Other temporary oilseed crops n.e.c.",
    "Olives",
    "Potatoes",
    "Tea",
    "Other beverage crops n.e.c.",
    "Chilies and peppers (capsicum spp.)",
    "Anise badian and fennel",
    "Other temporary spice crops n.e.c.",
    "Pepper (piper spp.)",
    "Beans",
    "Broad beans",
    "Chick peas",
    "Lentils",
    "Lupins",
    "Peas",
    "Leguminous crops n.e.c.",
    "Sugar beet",
    "Other sugar crops n.e.c.",
    "Grasses and other fodder crops",
    "Temporary grass crops",
    "Permanent grass crops",
    "Flax hemp and other similar products",
    "Temporary medicinal etc. crops",
    "Permanent medicinal etc. crops",
    "Flower crops",
    "Temporary flower crops",
    "Permanent flower crops",
    "Tobacco",
    "Other Classes",
    "Artificial Surfaces",
    "Fallow land",
    "Other crops",
    "Other crops temporary",
    "Other crops permanent",
    "Unknown crops"
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
    metric=['mIoU', 'mDice', 'mFscore'],
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
    frozen_backbone=True,
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
