import json
import torch

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_convert
from torchvision.transforms import v2

class HairDataset(Dataset):
    def __init__(self, images_path, annotations_path) -> None:
        super().__init__()
        self.images_path = images_path
        self.coco = COCO(annotation_file=annotations_path)

        with open(f'{annotations_path}') as file:
            self.raw_dictionary = json.load(file)

        self.list_of_image_dictionaries = self.raw_dictionary['images']

        # Adds all the annotations to the dictionary containing a particular images' details under the key 'list_of_annotations'
        for dict in self.list_of_image_dictionaries:
            dict['list_of_annotations'] = []
            for annotation in self.raw_dictionary['annotations']:
                if annotation['image_id'] == dict['id']:
                    dict['list_of_annotations'].append(annotation)
        
        self.image_transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()


    def __getitem__(self, index):
        """
        Returns a tuple containing 'Image' and 'Target'
        """
        image_path = f'{self.images_path}' + self.raw_dictionary['images'][index]['file_name']
        with Image.open(image_path) as pil_image:
            image = self.image_transforms(pil_image)
            image = image.float()
        target = {}
        target['area'] = []
        target['bbox'] = []
        target['image_id'] = torch.tensor(self.list_of_image_dictionaries[index]['id'])
        target['labels'] = []
        target['segmentation'] = []
        target['masks'] = []

        annotation_ids = self.coco.getAnnIds(imgIds=[self.list_of_image_dictionaries[index]['id']])
        annotations = self.coco.loadAnns(annotation_ids)
        mask_thing = self.coco.annToMask(annotations[0])

        for annotation in self.list_of_image_dictionaries[index]['list_of_annotations']:
            target['area'].append(annotation['area'])
            target['bbox'].append(annotation['bbox'])
            target['segmentation'].append(annotation['segmentation'])
            target['masks'].append(mask_thing)
            target['labels'].append(annotation['category_id'])
        
        target['masks'] = torch.tensor(target['masks'])
        
        target['area'] = torch.Tensor(target['area']).float()

        # Convert the boxes attribute to tensors and then format it to xyxy from xywh
        target['boxes'] = tv_tensors.BoundingBoxes(target['bbox'], format='xywh', canvas_size=(640, 640))
        target['boxes'] = box_convert(target['boxes'],  in_fmt='xywh', out_fmt='xyxy')

        target['labels'] = torch.Tensor(target['labels']).long()

        target.pop('segmentation', None)
        target.pop('bbox', None)

        return (image, target)
    
    def __len__(self):
        return len(self.raw_dictionary['images'])