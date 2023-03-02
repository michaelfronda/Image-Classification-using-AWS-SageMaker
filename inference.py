import subprocess
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests

def model_fn(model_dir):
    
    model = models.resnet50(pretrained = True) # pretrained 
    
    for param in model.parameters():
        param.required_grad = False # Convutional Layers = Freeze

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256), # first 
                             nn.ReLU(inplace = True),
                             nn.Linear(256, 133) # output should have 133 nodes as we have 133 classes of dog breeds
                            )
    
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    
    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object=transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction