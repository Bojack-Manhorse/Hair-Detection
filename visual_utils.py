import torch

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

def draw_masks_on_image(image:torch.Tensor, mask:torch.Tensor):
    booleans = [0.5 < item for item in mask]
    stacked = torch.stack(booleans)
    masks_summed = torch.sum(stacked, dim=0)
    masks_boolean = masks_summed.apply_(lambda x: x > 0).bool()
    new_masks = draw_segmentation_masks(image, masks_boolean)
    return to_pil_image(new_masks)