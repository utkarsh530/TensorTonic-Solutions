import torch

def patchify(images: list, patch_size: int) -> list:
    """
    Returns a float list of shape (B, num_patches, patch_size * patch_size * C).
    """
    images = torch.tensor(images, dtype = torch.float64)
    B,C,H,W = images.shape
    nh = H // patch_size
    nw = W // patch_size
    images = images.reshape(B,C, nh, patch_size, nw, patch_size)
    images = images.permute(0,2,4,3,5,1)
    images = images.reshape(B, nh * nw, C * patch_size * patch_size)
    return images.tolist()