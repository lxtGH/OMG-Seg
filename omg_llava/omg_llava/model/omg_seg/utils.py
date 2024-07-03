import torch
import torch.nn.functional as F

def mask_pool(x, mask):
    """
    Args:
        x: [B, C, H, W]
        mask: [B, Q, H, W]
    """
    if not x.shape[-2:] == mask.shape[-2:]:
        # reshape mask to x
        mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
    with torch.no_grad():
        mask = mask.detach()
        mask = (mask > 0).to(mask.dtype)
        denorm = mask.sum(dim=(-1, -2), keepdim=True) + 1e-8

    mask_pooled_x = torch.einsum(
        "bchw,bqhw->bqc",
        x,
        mask / denorm,
    )

    return mask_pooled_x



