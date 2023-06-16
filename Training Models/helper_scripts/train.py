'''
Trains a PyTorch image classification model using device agnostic code 
'''
import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer 
import argparse

# Setup some hyperparameters
num_epoch = 5
batch_size = 32
hidden_units = 10
learning_rate = 1e-3

# Setup directories
train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create DataLoader's and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=batch_size)

# Create model
model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


# Start the timer
start_time = timer()

# Start training with help from engine.py
engine.train(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, optimizer=optimizer, loss_fn=loss_fn, epochs=num_epoch, device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model to file
utils.save_model(model=model, target_dir='models', model_name='05_going_modular_script_mode_tinyvgg_model.pth')
