import numpy as np
from PIL import Image

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def expand2square_mask(mask):
    # mask (n, h, w)
    n_mask, width, height = mask.shape
    if width == height:
        return mask
    elif width > height:
        n_pad = width - height
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        pad_mask_1 = np.zeros((n_mask, width, n_pad_1), dtype=np.uint8)
        pad_mask_2 = np.zeros((n_mask, width, n_pad_2), dtype=np.uint8)
        result = np.concatenate([pad_mask_1, mask, pad_mask_2], axis=2)
        return result
    else:
        n_pad = height - width
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        pad_mask_1 = np.zeros((n_mask, n_pad_1, height), dtype=np.uint8)
        pad_mask_2 = np.zeros((n_mask, n_pad_2, height), dtype=np.uint8)
        result = np.concatenate([pad_mask_1, mask, pad_mask_2], axis=1)
        return result

def expand2square_bbox(bboxes, width, height):
    bboxes = np.array(bboxes)
    if width == height:
        return bboxes
    elif width > height:
        n_pad = width - height
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        bboxes[:, 1] += n_pad_1
        return bboxes
    else:
        n_pad = height - width
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        bboxes[:, 0] += n_pad_1
        return bboxes

def expand2square_points(points, width, height):
    if width == height:
        return points
    elif width > height:
        n_pad = width - height
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        points[:, 1] += n_pad_1
        return points
    else:
        n_pad = height - width
        n_pad_1 = n_pad // 2
        n_pad_2 = n_pad - n_pad_1
        points[:, 0] += n_pad_1
        return points

