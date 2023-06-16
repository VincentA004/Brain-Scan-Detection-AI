import torch
import torchvision
from torch import nn
from torchvision import transforms
import numpy as np
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
from cpc import cross_platform_compute
from helper_scripts import data_setup, engine, utils, helper_functions
from torchinfo import summary
from timeit import default_timer as timer

device = cross_platform_compute()


weights = torchvision.models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()

train_dir = '\\TSA\\Data\\Alzheimers\\train'
test_dir = '\\TSA\\Data\\Alzheimers\\test'
batch_size=32
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=auto_transforms, batch_size=batch_size)
print(class_names)
model = torchvision.models.vit_b_16(weights=weights).to(device)
 

# Freezing base layers
for param in model.parameters():
    param.requires_grad = False



'''
summary(
    model=model, 
    input_size=[1,3,224,224], # Example of [batch_size, color_channels, height, width]
    col_names=['input_size', 'output_size', 'num_params', 'trainable'], 
    col_width=20, 
    row_settings=['var_names']
)
'''


model.heads = nn.Sequential(
    nn.Linear(in_features=768, out_features=len(class_names))
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
epochs=500

start_time = timer()
# Setup training and save the results
results = engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, device=device)

# End the timer and print how long it took
end_time = timer()
print(f'[INFO] Total training time: {end_time-start_time:.3f}')

print(results) 
helper_functions.plot_loss_curves(results)
print(plt.show())

utils.save_model(model=model, target_dir='models', model_name=f'alzheimers_vit_b_16_{epochs}_epochs.pth')
