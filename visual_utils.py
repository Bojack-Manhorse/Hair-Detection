import torch

from PIL import Image
from torchvision.transforms.functional import to_pil_image

from torchvision.utils import draw_segmentation_masks

class DrawMasks():
    def __init__(self, model, transforms, device) -> None:
        self.transforms = transforms
        self.device = device
        self.model = model.to(device)
    
    def draw_masks_from_tensor_and_masks(self, image:torch.Tensor, mask:torch.Tensor, prob_threshold:float = 0.5):
        booleans = [prob_threshold < item for item in mask]
        stacked = torch.stack(booleans)
        masks_summed = torch.sum(stacked, dim=0).to('cpu')
        masks_boolean = masks_summed.apply_(lambda x: x > 0).bool().to(self.device)
        new_masks = draw_segmentation_masks(image, masks_boolean)
        return to_pil_image(new_masks)
    
    def draw_masks_from_tensor_and_model(self, image:torch.tensor, prob_threshold:float = 0.5):
        #image = image.unsqueeze(0)
        print(image.shape)
        mask = self.model(image)[0]['masks']
        return self.draw_masks_from_tensor_and_masks(image[0], mask, prob_threshold)
    
    def draw_mask_from_PIL_image_and_model(self, image:Image, prob_threshold:float = 0.5):
        image_tensor = self.transforms(image)
        image_tensor_unsqueezed = image_tensor.unsqueeze(0).to(self.device)
        return self.draw_masks_from_tensor_and_model(image_tensor_unsqueezed, prob_threshold)

    def draw_mask_from_image_path_and_model(self, path:str, prob_threshold:float = 0.5):
        with Image.open(path) as img:
            return self.draw_mask_from_PIL_image_and_model(img, prob_threshold)




"""
def draw_masks_on_image(image:torch.Tensor, mask:torch.Tensor, prob_threshold:float = 0.5):
    booleans = [prob_threshold < item for item in mask]
    stacked = torch.stack(booleans)
    masks_summed = torch.sum(stacked, dim=0).to('cpu')
    masks_boolean = masks_summed.apply_(lambda x: x > 0).bool().to(device)
    new_masks = draw_segmentation_masks(image, masks_boolean)
    return to_pil_image(new_masks)

def draw_mask_from_PIL_image(image:Image, model, transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(), prob_threshold:float = 0.5):
    image_tensor = transforms(image)
    image_tensor_unsqueezed = image_tensor.unsqueeze(0).to(device)
    model.eval()
    masks = model(image_tensor_unsqueezed)[0]['masks']
    return draw_masks_on_image(image_tensor_unsqueezed[0], masks, prob_threshold)

def draw_mask_from_image_path(path:str, model,transforms = MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(), prob_threshold:float = 0.5):
    with Image.open(path) as img:
        return draw_mask_from_PIL_image(img, model, transforms, prob_threshold)
"""