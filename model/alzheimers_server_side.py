import torch
import torchvision
from torch import nn
from timeit import default_timer as timer


def create_vit_model(num_classes:int=4, device='cpu'):
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights).to('cpu')
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.heads = nn.Sequential(
    nn.Linear(in_features=768, out_features=num_classes)
    )
    return model, transform



def predict(img):
    model_path = 'models/alzheimers_vit_b_16_500_epochs.pth'
    model, transform = create_vit_model()
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    
    start_time = timer()
    img = transform(img).unsqueeze(0) # adds batch dimension on 0th dimension
    
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]*100) for i in range(len(class_names))}
    
    end_time = timer()
    pred_time = round(end_time - start_time, 4)
    return pred_labels_and_probs, pred_time
