"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.nn.functional as NF
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sys
import os
import cv2
import copy
import glob

from models.model import *
from models.refinement_net import RefineModel
from models.modules import *
from utils import *
from visualize_utils import *
from evaluate_utils import *
from plane_utils import *
from options import parse_args
from config import InferenceConfig
from pathlib import Path
from utils import *
from datasets.plane_dataset import *
import json
from argparse import Namespace


class PlaneRCNN(nn.Module):
    def __init__(self, args):
        option_file = Path(__file__).parent / 'options.json'

        with open(option_file, 'r') as f:
            options_dict = json.load(f)
        self.options = Namespace(**vars(options_dict))
        config = InferenceConfig(self.options)
        self.config = config
        self.detector = PlaneRCNNDetector(self.options, self.config, 'final')

        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                config.BACKBONE_SHAPES,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)
        return

    def forward(self, Ks, images):
        normals = []
        masks = []
        for K, image in zip(Ks, images):
            H, W = images.shape[-2:]
            sample = self.preproces_input(Ks, images)
            res = self.detector(sample)[0]
            min_y = sample[1][0, 4]
            min_x = sample[1][0, 5]
            max_y = sample[1][0, 6]
            max_x = sample[1][0, 7]

            plane_normals = NF.normalize(res['detection'][:, 6:9], p=2, dim=-1).view(-1, 3, 1, 1)
            plane_normals = plane_normals[:, [0, 2, 1]]
            plane_normals[:, 0] *= -1
            merged_normal = (res['masks'].unsqueeze(1) * plane_normals).sum(0, keepdim=True)
            mask = res['mask'].unsqueeze(1)
            crop_normal = merged_normal[..., min_y:max_y, min_x:max_y]
            crop_mask = mask[..., min_y:max_y, min_x:max_y]

            full_normal_image = NF.interpolate(crop_normal, (H, W), mode='nearest')
            full_mask = NF.interpolate(crop_mask, (H, W), mode='nearest')
            normals.append(full_normal_image)
            masks.append(full_mask)
        return torch.cat(normals), torch.cat(masks)
    
    def preprocess_input(self, Ks, images):
        dev = images.device
        H, W = images.shape[-2:]
        fx = Ks[0, 0, 0]
        fy = Ks[0, 1, 1]
        cx = Ks[0, 0, 2]
        cy = Ks[0, 1, 2]
        index = 0
        camera = torch.Tensor([fx, fy, cx, cy, W, H], device=dev)
        image = NF.interpolate(images, (480, 640))
        extrinsics = np.eye(4, dtype=np.float32)

        ## The below codes just fill in dummy values for all other data entries which are not used for inference. You can ignore everything except some preprocessing operations on "image".
        depth = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.float32)
        segmentation = np.zeros((self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM), dtype=np.int32)


        planes = np.zeros((segmentation.max() + 1, 3))

        instance_masks = []
        class_ids = []
        parameters = []

        plane_offsets = np.linalg.norm(planes, axis=-1)
        plane_normals = planes / np.maximum(np.expand_dims(plane_offsets, axis=-1), 1e-4)
        distances_N = np.linalg.norm(np.expand_dims(plane_normals, 1) - self.config.ANCHOR_NORMALS, axis=-1)
        normal_anchors = distances_N.argmin(-1)

        for planeIndex, plane in enumerate(planes):
            m = segmentation == planeIndex
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(normal_anchors[planeIndex] + 1)
            normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[normal_anchors[planeIndex]]
            parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))

        parameters = np.array(parameters)
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(self.config, index, image, depth, mask, class_ids, parameters, augment=False)
        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]
            pass

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0).astype(np.float32)
        segmentation = np.concatenate([np.full((80, 640), fill_value=-1), segmentation, np.full((80, 640), fill_value=-1)], axis=0).astype(np.float32)

        data_pair = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match.astype(np.int32), rpn_bbox.astype(np.float32), gt_class_ids.astype(np.int32), gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32), gt_parameters[:, :-1].astype(np.float32), depth.astype(np.float32), extrinsics.astype(np.float32), planes.astype(np.float32), segmentation.astype(np.int64), gt_parameters[:, -1].astype(np.int32)]
        data_pair = data_pair + data_pair

        data_pair.append(np.zeros(7, np.float32))

        data_pair.append(planes)
        data_pair.append(planes)
        data_pair.append(np.zeros((len(planes), len(planes))))
        data_pair.append(camera.astype(np.float32))
        return data_pair

class PlaneRCNNDetector():
    def __init__(self, options, config, modelType):
        self.options = options
        self.config = config
        self.modelType = modelType
        self.model = MaskRCNN(config)
        self.model.to(options.device)
        self.model.eval()
        ckpt_dir = Path(__file__) / 'checkpoint' / 'planercnn_normal_warping_refine'
        self.model.load_state_dict(torch.load(ckpt_dir / 'checkpoint.pth', map_location=options.device))

        self.refine_model = RefineModel(options)
        self.refine_model.to(options.device)
        self.refine_model.eval()
        state_dict = torch.load(ckpt_dir / 'checkpoint_refine.pth', map_location=options.device)
        self.refine_model.load_state_dict(state_dict)

    def detect(self, sample):
        input_pair = []
        detection_pair = []
        dev = self.options.device
        camera = sample[30][0].to(self.options.device)
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].to(self.options.device), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].to(self.options.device), sample[indexOffset + 3].to(self.options.device), sample[indexOffset + 4].to(self.options.device), sample[indexOffset + 5].to(self.options.device), sample[indexOffset + 6].to(self.options.device), sample[indexOffset + 7].to(self.options.device), sample[indexOffset + 8].to(self.options.device), sample[indexOffset + 9].to(self.options.device), sample[indexOffset + 10].to(self.options.device), sample[indexOffset + 11].to(self.options.device)
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True)

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(self.config, camera, detections, detection_masks, depth_np_pred, debug=False)
                pass

            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
            detection_mask = detection_mask.unsqueeze(0)

            input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera})

            if 'nyu_dorn_only' in self.options.dataset:
                XYZ_pred[1:2] = sample[27].to(self.options.device)
                pass

            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'depth_np': depth_np_pred, 'plane_XYZ': plane_XYZ})
            continue

        if ('refine' in self.modelType or 'refine' in self.options.suffix):
            pose = sample[26][0].to(self.options.device)
            pose = torch.cat([pose[0:3], pose[3:6] * pose[6]], dim=0)
            pose_gt = torch.cat([pose[0:1], -pose[2:3], pose[1:2], pose[3:4], -pose[5:6], pose[4:5]], dim=0).unsqueeze(0)
            camera = camera.unsqueeze(0)

            for c in range(1):
                detection_dict, input_dict = detection_pair[c], input_pair[c]

                new_input_dict = {k: v for k, v in input_dict.items()}
                new_input_dict['image'] = (input_dict['image'] + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                new_input_dict['image_2'] = (sample[13].to(self.options.device) + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                detections = detection_dict['detection']
                detection_masks = detection_dict['masks']
                depth_np = detection_dict['depth_np']
                image = new_input_dict['image']
                image_2 = new_input_dict['image_2']
                depth_gt = new_input_dict['depth'].unsqueeze(1)

                masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)

                segmentation = new_input_dict['segmentation']

                detection_masks = torch.nn.functional.interpolate(detection_masks[:, 80:560].unsqueeze(1), size=(192, 256), mode='nearest').squeeze(1)
                image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
                image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
                masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
                depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                plane_depth = torch.nn.functional.interpolate(detection_dict['depth'][:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                segmentation = torch.nn.functional.interpolate(segmentation[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='nearest').squeeze().long()

                new_input_dict['image'] = image
                new_input_dict['image_2'] = image_2

                results = self.refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9], plane_depth, depth_np)

                masks = results[-1]['mask'].squeeze(1)

                all_masks = torch.softmax(masks, dim=0)

                masks_small = all_masks[1:]
                all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)
                all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).to(self.options.device).long().view((-1, 1, 1))).float()
                masks = all_masks[1:]
                detection_masks = torch.zeros(detection_dict['masks'].shape).to(self.options.device)
                detection_masks[:, 80:560] = masks


                detection_dict['masks'] = detection_masks
                detection_dict['depth_ori'] = detection_dict['depth'].clone()
                detection_dict['mask'][:, 80:560] = (masks.max(0, keepdim=True)[0] > (1 - masks.sum(0, keepdim=True))).float()

                if self.options.modelType == 'fitting':
                    masks_cropped = masks_small
                    ranges = self.config.getRanges(camera).transpose(1, 2).transpose(0, 1)
                    XYZ = torch.nn.functional.interpolate(ranges.unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1) * results[-1]['depth'].squeeze(1)
                    detection_areas = masks_cropped.sum(-1).sum(-1)
                    A = masks_cropped.unsqueeze(1) * XYZ
                    b = masks_cropped
                    Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                    AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                    plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) if detection_areas[planeIndex] else detection_dict['detection'][planeIndex, 6:9] for planeIndex in range(len(AA))], dim=0)
                    plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                    plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)
                    detection_dict['detection'][:, 6:9] = plane_parameters

                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detection_dict['detection'], detection_masks, detection_dict['depth'], return_individual=True)
                    detection_dict['depth'] = XYZ_pred[1:2]
                    pass
                continue
            pass
        return detection_pair

