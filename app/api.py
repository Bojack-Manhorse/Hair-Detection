import io
import torch
import torchvision
import uvicorn

from fastapi import FastAPI
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse, Response
from PIL import Image
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from visual_utils import DrawMasks

def get_model(weights_file = None):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer_size = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer_size,
        num_classes
    )
    if weights_file:
        state_dict = torch.load(weights_file)
        model.load_state_dict(state_dict)

    return model

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

try:
    model = get_model('../My Model.pt')
    model.eval()
    model.to(get_device())
    visualiser = DrawMasks(model, MaskRCNN_ResNet50_FPN_Weights.DEFAULT.transforms(), get_device())
except:
    raise OSError()

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
    msg = "API is up and running!"
    return {"message": msg}

@app.post('/get_mask')
def get_mask(image: UploadFile = File(...), prob_threshold:float = Form(...)):

    file_bytes = image.file.read()
    image = Image.open(io.BytesIO(file_bytes))
    image = visualiser.draw_mask_from_PIL_image_and_model(image, prob_threshold)
    bytes_image = io.BytesIO()
    image.save(bytes_image, format='PNG')

    return Response(content = bytes_image.getvalue(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run('api:app', host="0.0.0.0", port=8080)
