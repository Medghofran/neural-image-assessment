from torch import nn 
import torch
import torchvision.transforms as transforms
import torchvision as tv
import torchvision.models as models
from PIL import Image

from model import IMA 
import os

class NIMA(nn.Module):
    def __init__(self, base_model: nn.Module, input_features: int, drop_out: float):
        super(NIMA, self).__init__()
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(p=drop_out), nn.Linear(input_features, 10), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


MODELS = {
    "resnet18": (tv.models.resnet18, 512),
    "resnet34": (tv.models.resnet34, 512),
    "resnet50": (tv.models.resnet50, 2048),
    "resnet101": (tv.models.resnet101, 2048),
    "resnet152": (tv.models.resnet152, 2048),
    "resnet154": (tv.models.resnet152, 1280),
}


def create_model(model_type: str, drop_out: float) -> NIMA:
    create_function, input_features = MODELS[model_type]
    base_model = create_function(pretrained=True)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    return NIMA(base_model=base_model, input_features=input_features, drop_out=drop_out)

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SET UP ENCODER BACKBONE 
    base_model = models.vgg16(pretrained=True)
    # Inatialize image assessment class with created backbone
    model = create_model('resnet154', 0.1)
    # model = model.to(device)

    # LOAD PRETRAINED PARAMETERS
    pretrained_model_weights = './pretrained_models/pretrain-model.pth'
    model.load_state_dict(torch.load(pretrained_model_weights, map_location=torch.device("cpu")))

    # SET MODEL TO EVAULATION MODE
    model.eval()

    # PHOTO TRANSFORMS 
    test_transform = transforms.Compose([
    transforms.Scale(256), 
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
    ])
    
    # RUN INFERENCE OVER ONLY ONE IMAGE 
    im = Image.open(os.path.join('./data/images/10008.jpg'))
    im = im.convert('RGB')
    imt = test_transform(im)
    
    imt = imt.unsqueeze(dim=0)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    # print(out)
