import io
from model import get_model, get_device, load_model_weights
import torch
import uvicorn

from fastapi import FastAPI
from fastapi import File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from visual_utils import DrawMasks

try:
    """
    Attempt to load the model and cast it to the correct device.
    """
    model = get_model(2)
    model.eval()
    model.to(get_device())
except:
    raise OSError("Could not load model.")

try:
    """
    Attempt to load the model weights.
    """
    load_model_weights(model, 'Model_Weights.pt')
except:
    raise OSError("Could not load model weights.")

try:
    """
    Attempt to load an instance of the visualiser class.
    """
    visualiser = DrawMasks(model, MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(), get_device())
except:
    raise OSError("Could not create an instance of the visualiser class.")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"
    return {"message": msg}

@app.post('/get_mask')
def get_mask(image: UploadFile = File(...), prob_threshold:float = Form(...), resize_image:bool = Form(...)):
    """
    Function to get the masks from the model predictions of a particular image.

    Args:
        image: The image you wish to create the masks of.
        prob_threshold: The cut off point of the model's predictions to consider a pixel a part of the mask. Higher values mean a more conservative mask.
        resize_image: Whether or not to resize the image, so that the longest side is of length 1000 pixels.
    """

    # Read the image file
    file_bytes = image.file.read()

    # Convert the image to PIL Image format.
    image = Image.open(io.BytesIO(file_bytes))

    # Resize the image if the option was chosen, using a static method of the visualiser class.
    if resize_image:
        image = visualiser.resize_image(image)
    
    # Attempt to predict and draw the masks from image, using the 'draw_mask_from_PIL_image_and_model' method of the visualiser class.
    try:
        image = visualiser.draw_mask_from_PIL_image_and_model(image, prob_threshold)
    except torch.OutOfMemoryError:
        print("Ran out of memory, resizing image, try reszing the image.")
    
    # Create an instance of the io.BytesIO() class to load the masked image onto.
    bytes_image = io.BytesIO()

    # Load the image onto bytes_image.
    image.save(bytes_image, format='PNG')

    # Return the result.
    return Response(content = bytes_image.getvalue(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run('api:app', host="0.0.0.0", port=8080)
