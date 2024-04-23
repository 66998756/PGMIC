# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from mmseg.models.uda.teacher_module import EMATeacher
from mmseg.models.utils.dacs_transforms import get_mean_std, strong_transform
from mmseg.models.utils.masking_transforms import build_mask_generator

from mmseg.models.utils.masking_transforms import DualMaskGenerator, ClassMaskGenerator, SceneMaskGenerator

# 一些嘗試
from mmseg.models.utils.randaugment import RandAugment


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(MaskingConsistencyModule, self).__init__()

        self.source_only = cfg.get('source_only', False)
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength'] # 0.2
        self.color_jitter_p = cfg['color_jitter_probability'] # 0.2

        self.mask_mode = cfg['mask_mode']
        self.mask_alpha = cfg['mask_alpha']
        self.mask_pseudo_threshold = cfg['mask_pseudo_threshold']
        self.mask_lambda = cfg['mask_lambda']
        
        self.mask_gen = build_mask_generator(cfg['mask_generator'])

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        if require_teacher or \
                self.mask_alpha != 'same' or \
                self.mask_pseudo_threshold != 'same':
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

        # 一些嘗試
        self.consistency_mode = cfg['consistency_mode']
        if self.consistency_mode == 'fixmatch_like':
            self.strong_augment = RandAugment(0, 30)
        self.consistency_lambda = cfg['consistency_lambda']

        if cfg['consistency_lambda'] != None:
            self.lambda_weight = [self.mask_lambda, self.consistency_lambda]

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None,
                 local_hint_ratio=0.0):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(
                        target_img, target_img_metas, valid_pseudo_mask)
                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'Img': target_img.detach(),
                        'Pseudo Label': masked_plabel.cpu().numpy(),
                        'Pseudo Weight': masked_pweight.cpu().numpy(),
                    }
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
        else:
            raise NotImplementedError(self.mask_mode)

        ### Apply color augmentation
        # 參照 UniMatch 可以修改成 cutmix
        # 注意在 strong_transform 中 ColorJitter 參數都是相同的 self.color_jitter_s
        # ColorJitter(brightness, contrast, saturation, hue)
        # UniMatch 是 [0.5, 0.5, 0.5, 0.25]，接著做RandomGrayscale(p=0.2)
        # blur prob.=0.5, sigma=random.uniform(0.1, 2.0)
        # MIC 的 blur 單純是機率， sigma = np.random.uniform(0.15, 1.15)
        # 最後做 cutmix (見unimatch) (應該是相同的圖做兩個不同程度的aug後cutmix)
        if 'aug' in self.mask_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(
                strong_parameters, data=masked_img.clone())
        
        # 一些嘗試，應該要額外寫成API
        if self.consistency_mode == 'unimatch_like':
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': {
                    'brightness': 0.5,
                    'contrast': 0.5,
                    'saturation': 0.5,
                    'hue': 0.25,
                },
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            auged_img, _ = strong_transform(
                strong_parameters, data=target_img.clone(), mode='unimatch')
        elif self.consistency_mode == 'fixmatch_like':
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            auged_img, _ = strong_transform(
                strong_parameters, data=masked_img.clone())
            auged_img = self.strong_augment(auged_img)
            


        # Apply masking to image
        pseudo_label_region = valid_pseudo_mask.clone().bool()
        masked_imgs, mask_targets, hint_patch_nums = self.mask_gen.mask_image(
            masked_img, masked_lbl, pseudo_label_region, local_hint_ratio, mix=auged_img)

        # Train on masked images
        masked_losses = []
        for idx, masked_img in enumerate(masked_imgs):
            masked_losses.append(model.forward_train(
                masked_img,
                img_metas,
                masked_lbl,
                seg_weight=masked_seg_weight,
            ))
            
            # if self.debug and len(self.lambda_weight) > 1:
            #     self.debug_output['Masked_{}'.format(idx)] = model.debug_output
            #     if masked_seg_weight is not None:
            #         self.debug_output['Masked_{}'.format(idx)]['PL Weight'] = \
            #             masked_seg_weight.cpu().numpy()
            if self.debug and self.consistency_mode == 'unimatch_like':
                self.debug_output['Fused' if idx % 2 else 'Masked'] = model.debug_output
                if masked_seg_weight is not None:
                    self.debug_output['Fused' if idx % 2 else 'Masked']['PL Weight'] = \
                        masked_seg_weight.cpu().numpy()

        # Class and scene consistency
        # if isinstance(self.mask_gen, DualMaskGenerator):
        if len(masked_losses) > 1:
            for i, weight in enumerate(self.lambda_weight):
                for key in masked_losses[i].keys():
                    masked_losses[i][key] *= weight
        
            masked_loss = {}
            for key, value in masked_losses[0].items():
                masked_loss[key] = value
            
            # if self.mask_lambda != 1:
            #     masked_loss['decode.loss_seg'] *= self.mask_lambda

            # if self.debug:
            #     self.debug_output['Masked'] = model.debug_output
            #     if masked_seg_weight is not None:
            #         self.debug_output['Masked']['PL Weight'] = \
            #             masked_seg_weight.cpu().numpy()

            return masked_loss, mask_targets, hint_patch_nums
    
        # original MIC
        else:
            masked_loss = masked_losses[0]

            if self.mask_lambda != 1:
                masked_loss['decode.loss_seg'] *= self.mask_lambda

            if self.debug:
                self.debug_output['Masked'] = model.debug_output
                if masked_seg_weight is not None:
                    self.debug_output['Masked']['PL Weight'] = \
                        masked_seg_weight.cpu().numpy()
            
            return masked_loss, mask_targets, hint_patch_nums
