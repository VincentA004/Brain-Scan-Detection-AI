'''
Contains functionality for creating PyTorch DataLoader's for image classification data
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os 

num_workers = os.cpu_count()

def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=num_workers):
    """ Creates training and testing DataLoaders.
    
    Takes in a training directory and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders
    
    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for the number of workers per DataLoaders.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    Where class_names is a list of the target classes.
    Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=path/to/train_dir, test_dir=path/to/test_dir, transform=transform, batch_size=32, num_workers=4)
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    # Get the class names
    class_names = train_data.classes
    
    # Turn images into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
    
