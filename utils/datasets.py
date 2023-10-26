import torch
from torchvision.datasets import VisionDataset, VOCDetection, CocoDetection
from torchvision.ops import box_convert

import cv2
import copy
import os
import albumentations as A
import numpy as np

from typing import Any, Callable, List, Optional, Tuple
from pycocotools import mask as coco_mask

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class VOCDetectionV2(VOCDetection):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        target = self.convert_target(target)

        if not isinstance(self.transform, A.core.composition.Compose): RuntimeError("[+] The transform compose must by an Albumentations's type.!")
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(img), bboxes=target['boxes'], labels=target['labels'])
            img = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        
        ## COCO vars
        target["image_id"] = torch.tensor([index])
        target["area"] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
        # suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((len(target['labels']),), dtype=torch.int64)
        
        return img, target

    def convert_target(self,target):
        label_type=['background','aeroplane',"Bicycle",'bird',"Boat","Bottle","Bus","Car","Cat","Chair",'cow',"Diningtable","Dog","Horse","Motorbike",'person', "Pottedplant",'sheep',"Sofa","Train","TVmonitor"]
        convert_labels={}
        for idx, x in enumerate(label_type):
            convert_labels[x.lower()]=idx
        
        boxes = []
        labels = []
        for obj in target['annotation']['object']:
            labels.append(convert_labels[obj['name'].lower()])
            boxes.append([float(obj['bndbox'][_str]) for _str in ['xmin','ymin','xmax','ymax']])
            
        new_target = {}
        new_target["boxes"] = boxes
        new_target["labels"] = labels
        
        return new_target
    
class CocoDetectionV2(CocoDetection):
    
    def _convert_coco_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
            masks.append(mask)
        if masks:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
        return masks

    def _has_only_empty_bbox(self, anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(self, anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)
    
    def _has_less_or_equal_bbox(self, anno):
        return all(((obj['bbox'][0] + obj['bbox'][2] <= obj['bbox'][0]) and (obj['bbox'][1] + obj['bbox'][3] <= obj['bbox'][1])) for obj in anno)

    min_keypoints_per_image = 15

    def _has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if self._has_only_empty_bbox(anno):
            return False
        # if x_max/y_max is less than or equal to x_min/y_min for bbox
        if self._has_less_or_equal_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if self._count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False
    
    def __init__(self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, annFile, transform)
        
        self.ids_n = []
        coco_a = copy.deepcopy(self.coco)
        for ds_idx, img_id in enumerate(self.ids):
            ann_ids = coco_a.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = coco_a.loadAnns(ann_ids)
            if self._has_valid_annotation(anno):
                self.ids_n.append(img_id)

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = cv2.imread(os.path.join(self.root, path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            id = self.ids_n[index]

            image = self._load_image(id)
            h, w, _ = image.shape

            target = self._load_target(id)

            n_target = dict()
            for t in target:
                for k,v in t.items():
                    if k not in n_target: n_target[k] = []
                    n_target[k].append(v)

            n_target["bbox"] = torch.as_tensor(n_target["bbox"], dtype=torch.float32)
            n_target["bbox"] = box_convert(n_target["bbox"], in_fmt='xywh', out_fmt='xyxy')

            n_target["masks"] = self._convert_coco_poly_to_mask(n_target['segmentation'], h, w)
            del n_target['segmentation']

            f_target = dict()
            if self.transform is not None:
                if not isinstance(self.transform, A.core.composition.Compose): RuntimeError("[+] The transform compose must by an Albumentations's type.!")
                transformed = self.transform(image=np.asarray(image), 
                                             bboxes=n_target['bbox'],
                                             masks=[m.numpy() for m in n_target['masks']],
                                             category_ids=n_target['category_id'])
                image = transformed['image']
                f_target["masks"] = torch.stack(transformed['masks']).type(torch.uint8)
                f_target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                f_target["labels"] = torch.as_tensor(transformed['category_ids'], dtype=torch.int64)
            else:
                f_target["masks"] = n_target['masks'].type(torch.uint8)
                f_target["boxes"] = torch.as_tensor(n_target['bbox'], dtype=torch.float32)
                f_target["labels"] = torch.as_tensor(n_target['category_id'], dtype=torch.int64)
            f_target["image_id"] = torch.tensor([index])
            f_target["area"] = torch.as_tensor(n_target['area'])
            f_target["iscrowd"] = torch.as_tensor(n_target['iscrowd'])

            return image, f_target

        except:
            return None, None

    def __len__(self) -> int:
        return len(self.ids_n)
    
class LVISDetection(VisionDataset):

    def _has_only_empty_bbox(self, anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(self, anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)
    
    def _has_less_or_equal_bbox(self, anno):
        return all(((obj['bbox'][0] + obj['bbox'][2] <= obj['bbox'][0]) and (obj['bbox'][1] + obj['bbox'][3] <= obj['bbox'][1])) for obj in anno)

    min_keypoints_per_image = 15

    def _has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if self._has_only_empty_bbox(anno):
            return False
        # if x_max/y_max is less than or equal to x_min/y_min for bbox
        if self._has_less_or_equal_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if self._count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False
    
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from lvis import LVIS

        self.lvis = LVIS(annFile)
        self.ids = list(sorted(self.lvis.get_img_ids()))
        
        ## Sanity check
        self.ids_n = []
        for img_id in self.ids:
            anno = self.lvis.load_anns(self.lvis.get_ann_ids(img_ids=[img_id]))
            if self._has_valid_annotation(anno):
                self.ids_n.append(img_id)
        

    def _load_image(self, id: int):
        path = self.lvis.load_imgs([id])[0]["coco_url"].split("/")[-1]
        img = cv2.imread(os.path.join(self.root, path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def _load_target(self, id: int) -> List[Any]:
        return self.lvis.load_anns(self.lvis.get_ann_ids(img_ids=[id]))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            id = self.ids_n[index]

            image = self._load_image(id)
            target = self._load_target(id)

            n_target = dict()
            for t in target:
                for k,v in t.items():
                    if k not in n_target: n_target[k] = []
                    n_target[k].append(v)

            n_target["bbox"] = torch.as_tensor(n_target["bbox"], dtype=torch.float32)
            n_target["bbox"] = box_convert(n_target["bbox"], in_fmt='xywh', out_fmt='xyxy')

            n_target["masks"] = torch.as_tensor(np.array([self.lvis.ann_to_mask(ann_i) for ann_i in target]), dtype=torch.uint8)
            del n_target['segmentation']

            f_target = dict()
            if self.transform is not None:
                if not isinstance(self.transform, A.core.composition.Compose): RuntimeError("[+] The transform compose must by an Albumentations's type.!")
                transformed = self.transform(image=np.asarray(image), 
                                             bboxes=n_target['bbox'],
                                             masks=[m.numpy() for m in n_target['masks']],
                                             category_ids=n_target['category_id'])
                image = transformed['image']
                f_target["masks"] = torch.stack(transformed['masks']).type(torch.uint8)
                f_target["boxes"] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                f_target["labels"] = torch.as_tensor(transformed['category_ids'], dtype=torch.int64)
            else:
                f_target["masks"] = n_target['masks'].type(torch.uint8)
                f_target["boxes"] = torch.as_tensor(n_target['bbox'], dtype=torch.float32)
                f_target["labels"] = torch.as_tensor(n_target['category_id'], dtype=torch.int64)
            f_target["image_id"] = torch.tensor([index])
            f_target["area"] = torch.as_tensor(n_target['area'])
            f_target["iscrowd"] = torch.zeros((len(f_target['labels']),), dtype=torch.int64)

            return image, f_target

        except:
            return None, None

    def __len__(self) -> int:
        return len(self.ids_n)