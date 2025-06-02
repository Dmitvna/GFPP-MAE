import random
import torch
import torch.nn as nn
import numpy as np
from timm.layers.helpers import to_3tuple
from lib.networks import patch_embed_layers
import torch.nn.functional as F
from functools import partial
import torch.fft as fft

__all__ = ["MAE3D"]


def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2],
                  patch_size[2])  # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size),
                                                  np.prod(patch_size) * C)  # [B,gh*gw*gd,ph*pw*pd*C]

    return x


def batched_shuffle_indices(batch_size, length, device):
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


def compute_gradients(t):
    grad_x = F.pad(t[:, :, 1:, :, :] - t[:, :, :-1, :, :], pad=[0, 0, 0, 0, 0, 1], mode='constant')
    grad_y = F.pad(t[:, :, :, 1:, :] - t[:, :, :, :-1, :], pad=[0, 0, 0, 1, 0, 0], mode='constant')
    grad_z = F.pad(t[:, :, :, :, 1:] - t[:, :, :, :, :-1], pad=[0, 1, 0, 0, 0, 0], mode='constant')
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    return grad_magnitude


def calculate_frequency_loss(img1, img2, mask, weights):
    # Perform 3D FFT
    fft_img1 = fft.fftn(img1, dim=(-3, -2, -1))
    fft_img2 = fft.fftn(img2, dim=(-3, -2, -1))
    # Compute magnitude and phase
    magnitude_img1, phase_img1 = torch.abs(fft_img1), torch.angle(fft_img1)
    magnitude_img2, phase_img2 = torch.abs(fft_img2), torch.angle(fft_img2)
    # Compute frequency domain loss
    loss = torch.mean(weights * mask * ((magnitude_img1 - magnitude_img2) ** 2 + (phase_img1 - phase_img2) ** 2))

    return loss


class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv4 = nn.Conv3d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose3d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=768, out_channels=384, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose3d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=768, out_channels=384, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose3d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv1 = nn.Conv3d(in_channels=768, out_channels=384, kernel_size=3, padding=1)

    def forward(self, layer1, layer2, layer3, layer4):
        x4 = layer4.view(layer4.size(0), 6, 6, 6, 384).permute(0, 4, 1, 2, 3)
        x4 = self.conv4(x4)
        x4 = self.deconv4(x4)

        x3 = layer3.view(layer3.size(0), 6, 6, 6, 384).permute(0, 4, 1, 2, 3)
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.conv3(x3)
        x3 = self.deconv3(x3)

        x2 = layer2.view(layer2.size(0), 6, 6, 6, 384).permute(0, 4, 1, 2, 3)
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.conv2(x2)
        x2 = self.deconv2(x2)

        x1 = layer1.view(layer1.size(0), 6, 6, 6, 384).permute(0, 4, 1, 2, 3)
        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv1(x1)

        x1 = x1.permute(0, 2, 3, 4, 1).reshape(layer1.size(0), 216, 384)
        return x1


class MAE3D(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        grid_size = args.grid_size

        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = grid_size

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.encoder_embed_dim,
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                         args.encoder_embed_dim,
                                                                         num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)

        embed_layer = getattr(patch_embed_layers, args.patchembed)
        self.encoder = encoder(patch_size=patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        self.decoder = decoder(embed_dim=args.decoder_embed_dim,
                               depth=args.decoder_depth,
                               num_heads=args.decoder_num_heads)

        self.feature_fusion_pos = FeatureFusion()
        norm_layer_pos = partial(nn.LayerNorm, eps=1e-6)
        self.norm_pos = norm_layer_pos(args.decoder_embed_dim)

        self.feature_fusion_feat = FeatureFusion()
        norm_layer_feat = partial(nn.LayerNorm, eps=1e-6)
        self.norm_feat = norm_layer_feat(args.decoder_embed_dim)

        self.rec_head = nn.Linear(args.decoder_embed_dim, self.out_chans, bias=True)
        self.pos_head = nn.Linear(args.decoder_embed_dim, int(np.prod(self.grid_size)), bias=True)
        self.feat_head_1 = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim, bias=True)
        self.feat_head_2 = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)

        self.encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)

        self.mask_token_1 = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))

        self.patch_norm = nn.LayerNorm(normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False)

        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # initialize encoder_to_decoder and mask token
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.xavier_uniform_(self.rec_head.weight)
        nn.init.xavier_uniform_(self.pos_head.weight)
        nn.init.xavier_uniform_(self.feat_head_1.weight)
        nn.init.xavier_uniform_(self.feat_head_2.weight)
        nn.init.normal_(self.mask_token_1, std=.02)

    def forward(self, img, smooth_label, return_image=False):
        args = self.args
        batch_size = img.size(0)
        in_chans = img.size(1)
        assert in_chans == args.in_chans
        out_chans = self.out_chans

        # Cropping sub-volume and resizing
        s_x, s_y, s_z = random.randint(0, 64), random.randint(0, 64), random.randint(0, 64)
        e_x, e_y, e_z = s_x + 32, s_y + 32, s_z + 32
        img2 = F.interpolate(img[:, :, s_x:e_x, s_y:e_y, s_z:e_z].detach(), size=(96, 96, 96), mode='trilinear')

        # Calculate the proportion of sub-volume
        a = torch.ones_like(img, device=img.device)
        b = torch.zeros_like(img2, device=img2.device)
        b[:, :, s_x:e_x, s_y:e_y, s_z:e_z] = torch.ones((32, 32, 32))
        a = patchify_image(a, 32)
        b = patchify_image(b, 32)
        a = torch.sum(a, dim=-1)
        b = torch.sum(b, dim=-1)
        feat_label = b / a

        # Image partition
        x1 = patchify_image(img, self.patch_size)
        x2 = patchify_image(img2, self.patch_size)

        # Compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # Generate batched shuffle indices
        shuffle_indices_1 = batched_shuffle_indices(batch_size, length, device=x1.device)
        unshuffle_indices_1 = shuffle_indices_1.argsort(dim=1)

        # Select and mask the input patches
        shuffled_x1 = x1.gather(dim=1, index=shuffle_indices_1[:, :, None].expand(-1, -1, out_chans))
        sel_x1 = shuffled_x1[:, :sel_length, :]

        # Select and mask the indices
        sel_indices_1 = shuffle_indices_1[:, :sel_length]
        msk_indices_1 = shuffle_indices_1[:, -msk_length:]

        # Select the position embedings accordingly
        sel_encoder_pos_embed_1 = self.encoder_pos_embed.expand(batch_size, -1, -1) \
            .gather(dim=1, index=sel_indices_1[:, :, None]
                    .expand(-1, -1, args.encoder_embed_dim))

        # Forward encoder & proj to decoder dimension
        sel_x1 = self.encoder(sel_x1, sel_encoder_pos_embed_1)
        sel_x1 = self.encoder_to_decoder(sel_x1)

        # Combine the selected tokens and mask tokens in the shuffled order
        all_x1 = torch.cat([sel_x1, self.mask_token_1.expand(batch_size, msk_length, -1)], dim=1)

        # Shuffle all the decoder positional encoding
        shuffled_decoder_pos_embed_1 = self.decoder_pos_embed.expand(batch_size, -1, -1) \
            .gather(dim=1, index=shuffle_indices_1[:, :, None]
                    .expand(-1, -1, args.decoder_embed_dim))
        # Add the shuffled positional embedings to encoder output tokens
        all_x1[:, 1:, :] += shuffled_decoder_pos_embed_1

        # Forward decoder
        all_x1, shallow_feats = self.decoder(all_x1)
        l1, l2, l3, l4 = shallow_feats[0][:, 1:, :], shallow_feats[1][:, 1:, :], \
                         shallow_feats[2][:, 1:, :], shallow_feats[3][:, 1:, :]

        l1 = l1.gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, args.decoder_embed_dim))
        l2 = l2.gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, args.decoder_embed_dim))
        l3 = l3.gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, args.decoder_embed_dim))
        l4 = l4.gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, args.decoder_embed_dim))

        all_x1_pos = self.feature_fusion_pos(l1, l2, l3, l4)
        all_x1_pos = self.norm_pos(all_x1_pos)
        all_x1_feat = self.feature_fusion_feat(l1, l2, l3, l4)
        all_x1_feat = self.norm_feat(all_x1_feat)

        all_x2 = self.encoder(x2, self.encoder_pos_embed.expand(batch_size, -1, -1))

        # Head
        rec_x1 = self.rec_head(all_x1)
        pos_x1 = self.pos_head(all_x1_pos)
        feat_x1 = self.feat_head_1(all_x1_feat)
        feat_x2 = self.feat_head_2(all_x2)

        # Reconstruction
        masked_x = torch.cat(
            [shuffled_x1[:, :sel_length, :], 0. * torch.ones(batch_size, msk_length, out_chans).to(x1.device)],
            dim=1).gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, out_chans))
        recon = rec_x1[:, 1:, :].gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, out_chans))
        recon = recon * (x1.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6) + x1.mean(dim=-1, keepdim=True)
        recon_image = recon.reshape(batch_size, *self.grid_size, *self.patch_size, in_chans)
        recon_image = recon_image.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch_size, in_chans,
                                                                          self.grid_size[0] * self.patch_size[0],
                                                                          self.grid_size[1] * self.patch_size[1],
                                                                          self.grid_size[2] * self.patch_size[2])
        mask = torch.cat([torch.zeros(batch_size, sel_length, out_chans).to(x1.device),
                          torch.ones(batch_size, msk_length, out_chans).to(x1.device)],
                         dim=1).gather(dim=1, index=unshuffle_indices_1[:, :, None].expand(-1, -1, out_chans))
        mask = mask.reshape(batch_size, *self.grid_size, *self.patch_size, in_chans)
        mask = mask.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch_size, in_chans,
                                                            self.grid_size[0] * self.patch_size[0],
                                                            self.grid_size[1] * self.patch_size[1],
                                                            self.grid_size[2] * self.patch_size[2])

        # Rec loss
        img_grid = compute_gradients(img.detach())
        img_grid = patchify_image(img_grid, self.patch_size)
        img_grid = torch.sum(img_grid, dim=-1)
        min_grid = torch.min(img_grid, dim=-1, keepdim=True)[0]
        max_grid = torch.max(img_grid, dim=-1, keepdim=True)[0]
        weights = (img_grid - min_grid) / (max_grid - min_grid)
        weights = weights.unsqueeze(-1).repeat(1, 1, 4096)
        weights = weights.reshape(batch_size, *self.grid_size, *self.patch_size, in_chans)
        weights = weights.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch_size, in_chans,
                                                                  self.grid_size[0] * self.patch_size[0],
                                                                  self.grid_size[1] * self.patch_size[1],
                                                                  self.grid_size[2] * self.patch_size[2])

        rec_loss = calculate_frequency_loss(img.detach(), recon_image, mask, weights)

        # Pos loss / Abs loss
        smooth_label = smooth_label.to(pos_x1.device).detach()
        smooth_label_1 = torch.gather(smooth_label.unsqueeze(0).repeat(batch_size, 1, 1), dim=1,
                                      index=msk_indices_1[:, :, None].expand(-1, -1, int(length)))
        pos_x1 = pos_x1.gather(dim=1, index=msk_indices_1[:, :, None].expand(-1, -1, int(length)))
        pos_loss = torch.mean(((-smooth_label_1 * F.log_softmax(pos_x1, dim=-1)).sum(-1)))

        feat_x2 = torch.mean(feat_x2[:, 1:, :], dim=1, keepdim=True)

        region_feats = feat_x1.reshape(batch_size, 6, 6, 6, args.decoder_embed_dim).permute(0, 4, 1, 2, 3)
        region_feats = self.pool(region_feats)
        region_feats = region_feats.permute(0, 2, 3, 4, 1)
        region_feats = region_feats.reshape(batch_size, -1, args.decoder_embed_dim)

        # Div loss
        region_num = region_feats.shape[1]
        sim_regions = torch.cosine_similarity(region_feats.unsqueeze(2),
                                              region_feats.unsqueeze(1), dim=-1)
        sim_label = torch.eye(region_num, device=x1.device).expand(batch_size, -1, -1)
        div_loss = torch.sum(torch.abs(sim_regions - sim_label)) / (region_num * (region_num - 1))

        # Feat loss / Rel loss
        sim = self.cos(feat_x2, region_feats.detach())
        feat_loss = -torch.mean(torch.log(torch.ones_like(feat_label) - torch.abs(feat_label - sim)))

        # Total loss
        loss = args.b1 * rec_loss + args.b2 * pos_loss + args.b3 * feat_loss + args.b4 * div_loss

        if return_image:
            return loss, x1.detach(), recon.detach(), masked_x.detach()
        else:
            return loss
