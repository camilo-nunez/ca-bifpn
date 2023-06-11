import torch
from torchvision.datasets import VOCDetection #https://pytorch.org/vision/0.15/_modules/torchvision/datasets/voc.html

import cv2
import albumentations as A
import numpy as np

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

class VOCDetectionV2(VOCDetection):

    def __getitem__(self, index: int, transform=None):
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