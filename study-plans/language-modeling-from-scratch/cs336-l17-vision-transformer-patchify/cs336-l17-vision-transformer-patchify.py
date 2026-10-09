import torch

def vision_transformer_patchify(
    images: torch.Tensor, patch_height: int, patch_width: int,
) -> dict:

    B, C, H, W = images.shape

    assert H % patch_height == 0
    assert W % patch_width == 0

    nH = H // patch_height
    nW = W // patch_width

    images = images.reshape(B, C, nH, patch_height, nW, patch_width)
    images = images.permute(0, 2, 4, 1, 3, 5)

    tokens = images.reshape(B, nH * nW, C * patch_height * patch_width)

    coordinates = [
        [i // nW, i % nW]
        for i in range(nH * nW)
    ]

    coordinates = torch.tensor(
        coordinates, dtype=torch.int64, device=images.device
    )

    return {
        "tokens": tokens,
        "coordinates": coordinates
    }