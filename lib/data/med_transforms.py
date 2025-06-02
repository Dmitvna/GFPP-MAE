from monai import transforms


def get_scratch_train_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),
            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=args.RandRotate90d_prob,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform


def get_mae_pretrain_transforms(args):
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform


def get_val_transforms(args):
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"],
                                            a_min=args.a_min,
                                            a_max=args.a_max,
                                            b_min=args.b_min,
                                            b_max=args.b_max,
                                            clip=True),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return val_transform


def get_post_transforms(args):
    if args.test:
        post_pred = transforms.Compose([transforms.EnsureType(),
                                        transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
        post_label = transforms.Compose([transforms.EnsureType(),
                                         transforms.AsDiscrete(to_onehot=args.num_classes)])
    else:
        post_pred = transforms.Compose([transforms.EnsureType(),
                                        transforms.AsDiscrete(argmax=True, to_onehot=args.num_classes)])
        post_label = transforms.Compose([transforms.EnsureType(),
                                         transforms.AsDiscrete(to_onehot=args.num_classes)])

    return post_pred, post_label
