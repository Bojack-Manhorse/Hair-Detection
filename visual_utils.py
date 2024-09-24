import torch

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_segmentation_masks

class DrawMasks():
    """
    Class that draws masks on images in either tensor or PIL image form.
    """
    def __init__(self, model, transforms, device) -> None:
        """
        Initialise the model, image transformations and torch device

        Args:
            model: The neural network model which outputs the masks given a tensor representing an image.
            transforms: The transformations applied to a PIL image in order to convert it into a tensor which can be fed into the model.
            device: The device torch runs on.
        """

        self.transforms = transforms
        self.device = device
        self.model = model.to(device)
    
    def draw_masks_from_tensor_and_masks(self, image:torch.Tensor, mask:torch.Tensor, prob_threshold:float = 0.5):
        """
        Draws the masks on the image and returns the image in PIL format.

        Args:
            image: A tensor of shape (channels, height, width), where channels is the number of color channels of the image (usually 3).
            mask: A tensor of shape (num_masks, height, width), represeting num_masks different masks.
            prob_threshold: The cutoff for deciding if a pixel is within a mask or not based of its value in mask.
        
        Returns:
            A PIL image of the masks drawn on the image tensor.
        """
        
        # Take the max value for each pixel across all masks
        max_masks = torch.max(mask, dim=0, keepdim=False)[0]

        # Check if the max values are above the probability threshold
        booleans = torch.gt(max_masks, torch.tensor(prob_threshold))

        # Draw the masks on the image
        new_masks = draw_segmentation_masks(image, booleans)

        # Convert the tensor of the combined masks and images to a PIL image.
        return to_pil_image(new_masks)
    
    def draw_masks_from_tensor_and_model(self, image:torch.tensor, prob_threshold:float = 0.5):
        """
        Calcualates the mask predictions of an image tensor using the classes model and draws the masks on the image.

        Args:
            image: A tensor of shape (1, channels, height, width). The 0-th dimension is there since the model must take in batches of images, not single ones, so we feed it a batch of size one.
            prob_threshold: The cutoff for deciding if a pixel is within a mask or not based of its value in mask.
        
        Returns:
            A PIL image of the masks drawn on the image tensor.
        """

        # Set the model to eval mode.
        self.model.eval()
        
        # Get the mask output from the model
        mask = self.model(image)[0]['masks']

        # Draw the masks on the image using the method 'draw_masks_from_tensor_and_masks'
        return self.draw_masks_from_tensor_and_masks(image[0], mask, prob_threshold)
    
    def draw_mask_from_PIL_image_and_model(self, image:Image, prob_threshold:float = 0.5):
        """
        Calculates and draws the mask predictions of a PIL image (instead of a torch tensor)

        Args:
            image: A single PIL image
            prob_threshold: The cutoff for deciding if a pixel is within a mask or not based of its value in mask.
        
        Returns:
            A PIL image of the masks drawn on the image tensor.
        """

        # Turns the PIL image into a tensor using self.transforms.
        image_tensor = self.transforms(image)

        # Unsqueeze the tensor and cast it to the corret device so it can be fed into self.model.
        image_tensor_unsqueezed = image_tensor.unsqueeze(0).to(self.device)

        # Apply the 'draw_masks_from_tensor_and_model' method to get the image with masks.
        return self.draw_masks_from_tensor_and_model(image_tensor_unsqueezed, prob_threshold)

    def draw_mask_from_image_path_and_model(self, path:str, prob_threshold:float = 0.5):
        """
        Calculates and draws the mask predictions of the image located at 'path'.

        Args:
            path: A string represeting the path to a particular image.
            prob_threshold: The cutoff for deciding if a pixel is within a mask or not based of its value in mask.
        
        Returns:
            A PIL image of the masks drawn on the image tensor.
        """

        # Open the image at 'path' using the PIL.Image context manager.
        with Image.open(path) as img:

            # Return the masked image via the method 'draw_mask_from_PIL_image_and_model'.
            return self.draw_mask_from_PIL_image_and_model(img, prob_threshold)
    
    @staticmethod
    def resize_image(image:Image) -> Image:
        """
        Resizes an image so that the longest side is of length 1000

        Args:
            image: A PIL image that we wish to resize.
        
        Returns:
            The resized PIL image.
        """

        # Get the width an length of the image.
        width, length = image.size

        # Get the maximum of the two.
        max_dim = max((width, length))

        # Calculate the ratio to get the longer side to 1000.
        ratio = float(1000) / max_dim

        # Resize the image
        image = image.resize((int(width * ratio), int(length * ratio)))

        return image