from torchvision import models
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = models.get_model("maskrcnn_resnet50_fpn_v2", weights="MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT")

if __name__ == '__main__':
    pass