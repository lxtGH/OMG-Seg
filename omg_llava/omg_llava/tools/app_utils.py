import torch.nn.functional as F
import numpy as np
import torch

markdown_default = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
<style>
        .highlighted-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            color: rgb(255, 255, 239);
            background-color: rgb(225, 231, 254);
            border-radius: 7px;
            padding: 5px 7px;
            display: inline-block;
        }
        .regular-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 400;
            font-size: 14px;
        }
        .highlighted-response {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            border-radius: 6px;
            padding: 3px 4px;
            display: inline-block;
        }
</style>
<span class="highlighted-text" style='color:rgb(107, 100, 239)'>OMG-LLaVA</span>
"""

description = """
**Usage** : <br>
&ensp;(1) For **Grounded Caption Generation** Interleaved Segmentation, input prompt like: *"Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer."* <br>
&ensp;(2) For **Segmentation Output**, input prompt like: *"Can you please segment xxx in the given image"* <br>
&ensp;(3) For **Image Captioning** VQA, input prompt like: *"Could you please give me a detailed description of the image?"* <br>
&ensp;(4) For **Image Conversation**, input arbitrary text instruction. <br>
&ensp;(5) For **Visual prompt description**, first draw point or box in the image, and insert \<point\> or \<box\> into the text instruction. The input prompt like: *"Can you provide me with a detailed description of the region in the picture marked by region1 \<point\>?"* <br>
"""

ONE_THIRD = 1.0/3.0
ONE_SIXTH = 1.0/6.0
TWO_THIRD = 2.0/3.0

def desaturate(rgb, factor=0.65):
    """
    Desaturate an RGB color by a given factor.

    :param rgb: A tuple of (r, g, b) where each value is in [0, 255].
    :param factor: The factor by which to reduce the saturation.
                   0 means completely desaturated, 1 means original color.
    :return: A tuple of desaturated (r, g, b) values in [0, 255].
    """
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = rgb_to_hls(r, g, b)
    l = factor
    new_r, new_g, new_b = hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))

def rgb_to_hls(r, g, b):
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = (maxc+minc)
    rangec = (maxc-minc)
    l = sumc/2.0
    if minc == maxc:
        return 0.0, l, 0.0
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0-sumc)
    rc = (maxc-r) / rangec
    gc = (maxc-g) / rangec
    bc = (maxc-b) / rangec
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, l, s

def hls_to_rgb(h, l, s):
    if s == 0.0:
        return l, l, l
    if l <= 0.5:
        m2 = l * (1.0+s)
    else:
        m2 = l+s-(l*s)
    m1 = 2.0*l - m2
    return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))

def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < ONE_SIXTH:
        return m1 + (m2-m1)*hue*6.0
    if hue < 0.5:
        return m2
    if hue < TWO_THIRD:
        return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
    return m1

def process_markdown(output_str, colors):
    output_str = output_str.replace("\n", "").replace("  ", " ").replace("<s>", "")\
        .replace("<|im_end|>", '')
    output_str = output_str.split("ASSISTANT: ")[-1]

    markdown_out = output_str.replace('[SEG]', '')
    markdown_out = markdown_out.replace(
        "<p>", "<span class='highlighted-response' style='background-color:rgb[COLOR]'>"
    )
    markdown_out = markdown_out.replace("</p>", "</span>")

    for color in colors:
        markdown_out = markdown_out.replace("[COLOR]", str(desaturate(tuple(color))), 1)

    markdown_out = f""" 
    {markdown_out}
    """
    markdown_out = markdown_default + "<p><span class='regular-text'>" + markdown_out
    return markdown_out

def show_mask_pred(image, masks, crop_range=(0, 1024, 0, 1024)):
    print(crop_range)

    selected_colors = []

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255), [255, 192, 203],  # Pink
              [165, 42, 42],    # Brown
              [255, 165, 0],    # Orange
              [128, 0, 128],     # Purple
              [0, 0, 128],       # Navy
              [128, 0, 0],      # Maroon
              [128, 128, 0],    # Olive
              [70, 130, 180],   # Steel Blue
              [173, 216, 230],  # Light Blue
              [255, 192, 0],    # Gold
              [255, 165, 165],  # Light Salmon
              [255, 20, 147],   # Deep Pink
              ]

    masks = F.interpolate(masks, size=image.size, mode='bilinear', align_corners=False)
    masks = masks.sigmoid() > 0.5
    masks = masks.to(torch.uint8).cpu().numpy()[:, 0]

    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        selected_colors.append(color)
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]


    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = image[crop_range[2]: crop_range[3], crop_range[0]: crop_range[1], :]
    return image, selected_colors

def parse_visual_prompts(points):
    ret = {'points': [], 'boxes': []}
    for item in points:
        if item[2] == 1.0:
            ret['points'].append([item[0], item[1]])
        elif item[2] == 2.0 or item[2] == 3.0:
            ret['boxes'].append([item[0], item[1], item[3], item[4]])
        else:
            raise NotImplementedError
    return ret