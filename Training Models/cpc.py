
def cross_platform_compute():
    from sys import platform
    import torch
    
    if platform == "win32":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif platform == "darwin":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = 'cpu'
        
    return device


 