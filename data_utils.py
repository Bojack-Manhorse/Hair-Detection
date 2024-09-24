import json
import torch

from model import get_device
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_convert
from torchvision.transforms import v2

class HairDataset(Dataset):
    """
    A class for managing object segmentation datasets. Inherits from torch.utils.data.Dataset.
    """
    def __init__(self, folder_path:str, training_mode:bool = False) -> None:
        """
        Initialises the class.

        Args:
            folder_path: The path to the folder containing the images and annotations file.
            training_mode: If true, image augmentations (rotation, flips etc.) will be applied to the images.
        """
        super().__init__()

        # Set the folder path
        self.folder_path = folder_path

        # Set the training mode
        self.training_mode = training_mode

        # Set the annotations file, standard across this type of dataset.
        self.annotations_file = folder_path + '/_annotations.coco.json'

        # Initialise an instance of the COCO class for object segmentation. This allows us to read the masks/bounding boxes defined in _annotations.coco.json.
        self.coco = COCO(annotation_file=self.annotations_file)

        # Extract the data from _annotations.coco.json and load it into a dictionary
        with open(f'{self.annotations_file}') as file:
            self.raw_dictionary = json.load(file)

        # Each image has a corresponding dictionary in _annotations.coco.json, here we create a list of all these dictinaries.
        self.list_of_image_dictionaries = self.raw_dictionary['images']

        # Adds all the annotations to the dictionary containing a particular images' details under the key 'list_of_annotations'.
        # Iterate through all images in the dataset.
        for dict in self.list_of_image_dictionaries:

            # Initialise an empty list of annotations for the image.
            dict['list_of_annotations'] = []

            # Iterate through all annotations in _annotations.coco.json.
            for annotation in self.raw_dictionary['annotations']:

                # If the annotation matches the image, add it to the dictionary corresponding to that particulate image.
                if annotation['image_id'] == dict['id']:
                    dict['list_of_annotations'].append(annotation)
        
        # Define the transformations applied to the image to turn it into an appropriate tensor (not including any image augmentations.)
        self.image_transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms()


    def __getitem__(self, index):
        """
        Returns a tuple containing 'image' and 'target'
        """
        device = get_device()

        # Initialise target to be an empty dictionary.
        target = {}

        # Initialise all the keys of target as empty lists.
        target['area'] = []
        target['bbox'] = []
        target['image_id'] = torch.tensor(self.list_of_image_dictionaries[index]['id'])
        target['labels'] = []
        target['segmentation'] = []
        target['masks'] = []
        target['iscrowd'] = []

        # Get all the annotations corresponding to the image at 'index' using 'self.list_of_image_dictionaries'.
        annotation_ids = self.coco.getAnnIds(imgIds=[self.list_of_image_dictionaries[index]['id']])

        # Iterate through all the annotations corresponding to an image to fill out the keys of target.
        for annotation in self.list_of_image_dictionaries[index]['list_of_annotations']:
            target['area'].append(annotation['area'])
            target['bbox'].append(annotation['bbox'])
            target['segmentation'].append(annotation['segmentation'])
            target['labels'].append(annotation['category_id'])
            target['iscrowd'].append(False)
            
        # Load all the annotations corresponding to the image as coco annotations, so we can extract the masks in the correct format.
        annotations = self.coco.loadAnns(annotation_ids)

        # Iterate through all the annotations and add thier masks to 'target[masks]'
        for item in annotations:
            mask_thing = torch.tensor(self.coco.annToMask(item))
            target['masks'].append(mask_thing)
        
        # Convert 'target['mask']' to 'torchvision.tv_tensors.Mask' format.
        target['masks'] = tv_tensors.Mask(torch.stack(target['masks'])).to(device)
        
        # Convert the area, labels and iscrowd keys to tensors in the correct device.
        target['area'] = torch.Tensor(target['area']).float().to(device)
        target['labels'] = torch.Tensor(target['labels']).long().to(device)
        target['iscrowd'] = torch.Tensor(target['iscrowd']).to(device)

        # Convert the boxes attribute to tensors and then format it to xyxy from xywh
        image_height = self.raw_dictionary['images'][index]['height']
        image_width = self.raw_dictionary['images'][index]['width']

        target['boxes'] = tv_tensors.BoundingBoxes(target['bbox'], format='xywh', canvas_size=(image_height, image_width))
        target['boxes'] = box_convert(target['boxes'],  in_fmt='xywh', out_fmt='xyxy').to(device)
        target['boxes'] = tv_tensors.BoundingBoxes(target['boxes'], format='xyxy', canvas_size=(image_height, image_width))

        # Remove unnecessary keys from target.
        target.pop('segmentation', None)
        target.pop('bbox', None)

        # Get the image path.
        image_path = f'{self.folder_path}/' + self.raw_dictionary['images'][index]['file_name']
        
        # Set the image transformations corresponding to augmentations if training_mode is true.
        if self.training_mode:
            transformers = v2.Compose([
                v2.RandomRotation(degrees=(0, 10)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ])
        else:
            transformers = v2.Compose([v2.RandomHorizontalFlip(p=0)])
        
        # Open the image as a PIL image and apply the transformations to it an target.
        with Image.open(image_path) as pil_image:
            image, target = transformers(pil_image, target)
            image = self.image_transforms(image)

        # Cast the image to the correct format and device
        image = image.float().to(device)

        return (image, target)
    
    def __len__(self):
        """
        Returns the lenght of the dataset.
        """
        return len(self.raw_dictionary['images'])