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
    elif t == 'scene':
        return SceneMaskGenerator(**cfg)
    elif t == 'dual':
        return DualMaskGenerator(**cfg)
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

        mask_targets = []
        for batch in range(B):
            current_classes = torch.unique(lbls[batch])
            # ignore "void" class
            # if torch.max(current_classes) == 255:
            void_mask = current_classes != 255
            current_classes = current_classes[void_mask]
            
            mask_target = current_classes[torch.randint(0, len(
                current_classes), (1,)).item()]
            # manually set mask target class to 3: wall
            # mask_target = 3
            mask_targets.append(mask_target.item())
            
            class_mask = (lbls[batch] == mask_target).float()

            unfolded_mask = torch.nn.functional.unfold(class_mask.unsqueeze(dim=0), 
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

            unfolded_block_mask = torch.any(unfolded_mask.bool(), dim=1)

            if torch.sum(unfolded_block_mask) / unfolded_block_mask.numel() < self.mask_ratio:
                remain_mask_times = int(
                    self.mask_ratio * unfolded_block_mask.numel() - torch.sum(unfolded_block_mask))
                
                zero_indices = torch.where(unfolded_block_mask == False)[1]
                random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

                unfolded_block_mask[:, zero_indices[random_indices]] = True

            block_mask = (~unfolded_block_mask.expand(unfolded_mask.shape)).float()
            
            input_mask[batch, :, :, :] = torch.nn.functional.fold(block_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

        return input_mask, mask_targets

    @torch.no_grad()
    def mask_image(self, imgs, lbls):
        input_mask, mask_targets = self.generate_mask(imgs, lbls)
        return imgs * input_mask, mask_targets


class SceneMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs, lbls):
        B, C, H, W = imgs.shape
        ### Implement of Class Masking ###
        input_mask = torch.zeros((B, 1, H, W), device=imgs.device)

        mask_targets = []
        for batch in range(B):
            current_classes = torch.unique(lbls[batch])
            # ignore "void" class
            void_mask = current_classes != 255
            current_classes = current_classes[void_mask]
            
            mask_target = current_classes[torch.randint(0, len(
                current_classes), (1,)).item()]
            # manually set mask target class to 3: wall
            # mask_target = 3
            mask_targets.append(mask_target.item())
            
            class_mask = (lbls[batch] == mask_target).float()

            unfolded_mask = torch.nn.functional.unfold(class_mask.unsqueeze(dim=0), 
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

            unfolded_block_mask = torch.any(unfolded_mask.bool(), dim=1)

            # only display choised target class and other random patch
            if torch.sum(unfolded_block_mask) / unfolded_block_mask.numel() < (1 - self.mask_ratio):
                remain_mask_times = int(
                    (1 - self.mask_ratio) * unfolded_block_mask.numel() - torch.sum(unfolded_block_mask))
                
                zero_indices = torch.where(unfolded_block_mask == False)[1]
                random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

                unfolded_block_mask[:, zero_indices[random_indices]] = True

            block_mask = (unfolded_block_mask.expand(unfolded_mask.shape)).float()
            
            input_mask[batch, :, :, :] = torch.nn.functional.fold(block_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

        return input_mask, mask_targets

    @torch.no_grad()
    def mask_image(self, imgs, lbls):
        input_mask, mask_targets = self.generate_mask(imgs, lbls)
        return imgs * input_mask, mask_targets
    

class DualMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs, lbls):
        B, C, H, W = imgs.shape
        ### Implement of Class Masking ###
        class_maskes = torch.zeros((B, 1, H, W), device=imgs.device)
        scane_maskes = torch.zeros((B, 1, H, W), device=imgs.device)

        mask_targets = []
        for batch in range(B):
            current_classes = torch.unique(lbls[batch])
            # ignore "void" class
            if torch.max(current_classes) == 255:
                mask = current_classes != 255
                current_classes = current_classes[mask]
            mask_target = current_classes[torch.randint(0, len(
                current_classes), (1,)).item()]
            # manually set mask target class to 1: sidewalk
            # mask_target = 1
            mask_targets.append(mask_target.item())
            
            class_mask = (lbls[batch] == mask_target).float()

            unfolded_mask = torch.nn.functional.unfold(class_mask.unsqueeze(dim=0), 
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

            unfolded_block_mask = torch.any(unfolded_mask.bool(), dim=1)

            # class mask
            if torch.sum(unfolded_block_mask) / unfolded_block_mask.numel() < self.mask_ratio:
                remain_mask_times = int(
                    self.mask_ratio * unfolded_block_mask.numel() - torch.sum(unfolded_block_mask))
                
                zero_indices = torch.where(unfolded_block_mask == False)[1]
                random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

                
                unfolded_class_mask = unfolded_block_mask.clone()
                unfolded_class_mask[:, zero_indices[random_indices]] = True
            else:
                unfolded_class_mask = unfolded_block_mask.clone()

            # scene mask
            if torch.sum(unfolded_block_mask) / unfolded_block_mask.numel() < (1 - self.mask_ratio):
                remain_mask_times = int(
                    (1 - self.mask_ratio) * unfolded_block_mask.numel() - torch.sum(unfolded_block_mask))
                
                zero_indices = torch.where(unfolded_block_mask == False)[1]
                random_indices = torch.randperm(len(zero_indices))[:remain_mask_times]

                unfolded_scane_mask = unfolded_block_mask.clone()
                unfolded_scane_mask[:, zero_indices[random_indices]] = True
            else:
                unfolded_scane_mask = unfolded_block_mask.clone()

            unfolded_class_mask = (~unfolded_class_mask.expand(unfolded_mask.shape).bool()).float()
            unfolded_scane_mask = (unfolded_scane_mask.expand(unfolded_mask.shape).bool()).float()
            
            class_maskes[batch, :, :, :] = torch.nn.functional.fold(unfolded_class_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)
            scane_maskes[batch, :, :, :] = torch.nn.functional.fold(unfolded_scane_mask, lbls.shape[2:],
                kernel_size=self.mask_block_size, stride=self.mask_block_size)

        return [class_maskes, scane_maskes], mask_targets

    @torch.no_grad()
    def mask_image(self, imgs, lbls):
        return_imgs = []
        input_maskes, mask_targets = self.generate_mask(imgs, lbls)
        for mask in input_maskes:
            return_imgs.append(imgs * mask)
        return return_imgs, mask_targets


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
        return imgs * input_mask, None
