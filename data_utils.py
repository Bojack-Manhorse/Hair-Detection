import json
import torch

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors, transforms
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_convert
from torchvision.transforms import v2

class HairDataset(Dataset):
    def __init__(self, folder_path, training_mode:bool = False) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.training_mode = training_mode
        self.annotations_file = folder_path + '/_annotations.coco.json'
        self.coco = COCO(annotation_file=self.annotations_file)

        with open(f'{self.annotations_file}') as file:
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
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        target = {}
        target['area'] = []
        target['bbox'] = []
        target['image_id'] = torch.tensor(self.list_of_image_dictionaries[index]['id'])
        target['labels'] = []
        target['segmentation'] = []
        target['masks'] = []
        target['iscrowd'] = []

        annotation_ids = self.coco.getAnnIds(imgIds=[self.list_of_image_dictionaries[index]['id']])

        for annotation in self.list_of_image_dictionaries[index]['list_of_annotations']:
            target['area'].append(annotation['area'])
            target['bbox'].append(annotation['bbox'])
            target['segmentation'].append(annotation['segmentation'])
            annotations = self.coco.loadAnns(annotation_ids)
            mask_thing = torch.tensor(self.coco.annToMask(annotations[0]))
            target['masks'].append(mask_thing)
            target['labels'].append(annotation['category_id'])
            target['iscrowd'].append(False)
            
        
        target['masks'] = torch.stack(target['masks']).to(device)
        
        target['area'] = torch.Tensor(target['area']).float().to(device)

        # Convert the boxes attribute to tensors and then format it to xyxy from xywh
        target['boxes'] = tv_tensors.BoundingBoxes(target['bbox'], format='xywh', canvas_size=(640, 640))
        target['boxes'] = box_convert(target['boxes'],  in_fmt='xywh', out_fmt='xyxy').to(device)

        target['labels'] = torch.Tensor(target['labels']).long().to(device)

        target['iscrowd'] = torch.Tensor(target['iscrowd']).to(device)

        target.pop('segmentation', None)
        target.pop('bbox', None)

        print([type(item) for item in target.values()])

        image_path = f'{self.folder_path}/' + self.raw_dictionary['images'][index]['file_name']


        if self.training_mode:
            transformers = v2.Compose([
                v2.RandomRotation(degrees=(0, 90)),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        
        with Image.open(image_path) as pil_image:
            #image = self.image_transforms(pil_image)
            image, target = transformers(pil_image, target)
            image = self.image_transforms(image)

        
        
        image = image.float().to(device)


        return (image, target)
    
    def __len__(self):
        return len(self.raw_dictionary['images'])