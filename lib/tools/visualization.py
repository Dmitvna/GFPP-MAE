import numpy as np
import torch
import torchvision
from timm.layers.helpers import to_3tuple


def patches3d_to_grid(patches, patch_size=16, grid_size=6, in_chans=1, n_group=3, hidden_axis='d',
                      slice_pos_list=[0.4, 0.45, 0.5, 0.55, 0.6]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --patches: [B, L, C*H*W*D]
    --patch_size: 
    --grid_size:
    --in_chans:
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, L, C = patches.shape   # [3, 216, 4096]
    patch_size = to_3tuple(patch_size)  # [16, 16, 16]
    grid_size = to_3tuple(grid_size)  # [6, 6, 6]
    assert np.prod(grid_size) == L and np.prod(patch_size) * in_chans == C, "Shape of input doesn't match parameters"

    patches = patches.reshape(B, *grid_size, *patch_size, in_chans)  # [3, 6, 6, 6, 16, 16, 16, 1]

    # restore image structure
    image = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, in_chans, 1,
                                                            grid_size[0] * patch_size[0],
                                                            grid_size[1] * patch_size[1],
                                                            grid_size[2] * patch_size[2])   # [3, 1, 1, 96, 96, 96]
    assert B % n_group == 0
    n_per_row = len(slice_pos_list) * in_chans * B // n_group
    # always choose the specified slice to visualize
    if hidden_axis == 'd':
        slice_list = []
        for slice_pos in slice_pos_list:
            slice_list.append(image[..., :, :, int(image.size(-1) * slice_pos)])
        image = torch.cat(slice_list, dim=2)   # [3, 1, 5, 96, 96], 只能可视化2D图像，选择3D图像的5个slice
    else:
        raise ValueError(f"Only support D for now")
    visH, visW = image.size(-2), image.size(-1)
    grid_of_images = torchvision.utils.make_grid(image.reshape(B * len(slice_pos_list) * in_chans, 1, visH, visW),
                                                 nrow=n_per_row, padding=0)
    grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()   # [3, 96*3, 96*5]

    return grid_of_images


def images3d_to_grid(image, n_group=3, hidden_axis='d', slice_pos_list=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """
    input patches are in 3D which contain height, width and depth
    -------
    Params:
    --image: [B, C, H, W, D]
    --n_groups: group number of patches, e.g., original patch group, masked patch group, recon patch group
    --hidden_axis: indicate the axis to be hidden because we can only visualize a 2D image instead of 3D volume
    """
    B, C, H, W, D = image.shape

    assert B % n_group == 0
    n_per_row = B // n_group
    list_of_grid_images = []
    for slice_pos in slice_pos_list:
        if hidden_axis == 'd':
            image_slice = image[..., :, :, int(D * slice_pos)]  # [B, 3, H, W]
        else:
            raise ValueError(f"Only support D for now")
        grid_of_images = torchvision.utils.make_grid(image_slice, nrow=n_per_row)
        grid_of_images.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        list_of_grid_images.append(grid_of_images)

    return list_of_grid_images
