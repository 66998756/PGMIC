# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch

from mmseg.ops import resize


def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'random':
        return RandomMaskGenerator(**cfg)
    elif t == 'class':
        return ClassMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class ClassMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs, lbls):
        B, C, H, W = imgs.shape
        ### Implement of Class Masking ###
        input_mask = torch.zeros((B, 1, H, W), device=imgs.device)

        target_classes = []
        for batch in range(B):
            current_classes = torch.unique(lbls[batch, :, :, :])
            random_mask_target_class = current_classes[torch.randint(0, len(
                current_classes), (1,)).item()]
            # manually set mask target class to 3: wall
            # random_mask_target_class = 3
            target_classes.append(random_mask_target_class)
            
            class_mask = (lbls[batch, :, :, :] == random_mask_target_class).float()

            unfolded_mask = torch.nn.functional.unfold(class_mask.unsqueeze(dim=0), 
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

            block_mask_indices = torch.any(unfolded_mask.bool(), dim=1)

            if torch.sum(block_mask_indices) / block_mask_indices.numel() < self.mask_ratio:
                remain_mask_times = int(
                    self.mask_ratio * block_mask_indices.numel() - torch.sum(block_mask_indices))
                
                zero_indices = torch.where(block_mask_indices == False)[1]
                random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

                block_mask_indices[:, zero_indices[random_indices]] = True

            block_mask = (~block_mask_indices.expand(unfolded_mask.shape)).float()
            # unfolded_mask = torch.where(block_mask_indices == 1,
            #     mask_block_label, non_mask_block_label)
            
            input_mask[batch, :, :, :] = torch.nn.functional.fold(block_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs, lbls):
        input_mask = self.generate_mask(imgs, lbls)
        return imgs * input_mask


# original MIC masking method
class RandomMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs, lbls):
        B, C, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs, lbls):
        input_mask = self.generate_mask(imgs, lbls)
        return imgs * input_mask
