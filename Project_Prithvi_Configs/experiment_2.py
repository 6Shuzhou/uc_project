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
pretrained_weights_path = "Pretrained_Prithvi_Weights\Prithvi_EO_V1_100M.pt" # Absolute Path to Pre Trained Weights, replace Spaces with %20
num_layers = 6
patch_size = 61
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 10
eval_epoch_interval = 1

loss_weights_multi = [
    #0,
    56.567317916002125,
    6.157588252314815,
    0,
    14.726057020275414,
    0.20686247063134944,
    32.94724415544202,
    0.8966568071022571,
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
    0.1833657746804118,
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
    17.672002159109784,
    61.844303981400756,
    0,
    25.234940116210126,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    4.84697073226284,
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
experiment = "Experiment_2"
project_dir = "UC_Project"
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


dataset_type = "GeospatialDataset"

# TO BE DEFINED BY USER: data directory
data_root = "C:/Users/Gebruiker/Documents/LU%20UC/Project/uc_project/Experiments_Transformed_Selected_Subset/Experiment_2" # Absolute Path to Data Root Directory, replace Spaces with %20

img_norm_cfg = dict(
    means=[
        1090.548095703125,
        1012.2404174804688,
        1027.209716796875,
        1101.2984619140625,
        1153.42041015625,
        1206.4273681640625,
        1199.9398193359375,
        1176.298828125,
        1124.452880859375,
        1388.6683349609375,
        1675.1827392578125,
        1997.4666748046875,
        1008.4226684570312,
        981.6507568359375,
        1035.444580078125,
        1116.1083984375,
        1161.5950927734375,
        1260.30859375,
        1261.944091796875,
        1241.2620849609375,
        1149.492431640625,
        1410.8675537109375,
        1693.901611328125,
        2016.471435546875,
    ],
    stds=[
        145.8170928955078,
        91.79450225830078,
        83.80763244628906,
        103.34722900390625,
        122.53270721435547,
        104.32010650634766,
        98.28654479980469,
        101.53093719482422,
        104.74579620361328,
        233.4324188232422,
        520.2266845703125,
        805.4223022460938,
        175.01368713378906,
        130.4468231201172,
        133.82000732421875,
        153.5885009765625,
        177.1007537841797,
        165.55931091308594,
        161.2931671142578,
        166.0906219482422,
        166.68118286132812,
        260.4395446777344,
        571.2723388671875,
        893.4560546875,
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
    "Mixed cereals",
    "Other",
    "Leafy or stem vegetables",
    "Artichokes",
    "Asparagus",
    "Cauliflowers and broccoli",
    "Lettuce",
    "Spinach",
    "Cucumbers",
    "Eggplants (aubergines)",
    "Tomatoes",
    "Watermelons",
    "Cantaloupes and other melons",
    "Pumpkin squash and gourds",
    "Root bulb or tuberous vegetables",
    "Carrots",
    "Onions (incl. shallots)",
    "Leeks and other alliaceous vegetables",
    "Mushrooms and truffles",
    "Figs",
    "Other tropical and subtropical fruits n.e.c.",
    "Grapefruit and pomelo",
    "Lemons and Limes",
    "Oranges",
    "Tangerines mandarins clementines",
    "Grapes",
    "Currants",
    "Kiwi fruit",
    "Strawberries",
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
    "Olives",
    "Potatoes",
    "Tea",
    "Other beverage crops n.e.c.",
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
    "Forest",
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
