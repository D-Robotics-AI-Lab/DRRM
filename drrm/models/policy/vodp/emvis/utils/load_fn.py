# Reference:
#   https://github.com/facebookresearch/vggt/blob/main/vggt/utils/load_fn.py

import logging
import torch
from PIL import Image
from torchvision import transforms
logger = logging.getLogger(__name__)

def load_and_preprocess_images(origin_images, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        origin_images (torch.Tensor): Batched tensor of original RGB images with shape (N, 3, H, W) 
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H', W')

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    # if len(image_path_list) == 0:
    #     raise ValueError("At least 1 image is required")
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    # TODO origin_images 转 Image
    for img in origin_images:
        width, height = img.size
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
        
        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    return images
MODE = "crop"
def get_resize_shape(height, width, target_size = None):
    target_size = width if target_size==None else target_size
    target_width = round(target_size / 14) * 14
    height = round(height*(target_width/width) / 14) * 14
    width = target_width
    # height = round(height / 14) * 14
    # width = round(width / 14) * 14
    # height = height // 14 * 14
    # width = width // 14 * 14

    return int(height), int(width)

def get_processed_shape(height, width, mode=MODE):
    height, width = get_resize_shape(height, width)
    if mode == "pad":
        max_size = round(max(width, height) // 14) * 14
        return (max_size, max_size)
    else:  # mode == "crop"
        new_width = (width // 14) * 14
        new_height = min((height // 14) * 14, new_width)
        return (new_height, new_width)

def preprocess_images(origin_images: torch.Tensor, mode=MODE, interpolate=None):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        origin_images (torch.Tensor): Batched tensor of original RGB images with shape (N, 3, H, W), in range [0, 1].
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    assert len(origin_images.shape) == 4, "The shape of images should be (N, 3, H, W)"
    N, C_in, H, W = origin_images.shape
    assert C_in == 3, "Only accept RGB images and the shape should be (N, 3, H, W)"
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()

    # First process all images and collect their shapes
    # TODO origin_images 转 Image
    if isinstance(interpolate, str):
        interpolate = (interpolate, W)
    elif isinstance(interpolate, int):
        interpolate = ('bilinear', interpolate)
    if interpolate:
        interstr, t_size = interpolate
        height, width = get_resize_shape(H, W, t_size if t_size else W)
        if interstr=='bicubic':
            intermod = transforms.InterpolationMode.BICUBIC
        elif interstr=='bilinear':
            intermod = transforms.InterpolationMode.BILINEAR
        elif interstr=='nearest':
            intermod = transforms.InterpolationMode.NEAREST
        else:
            raise ValueError("Only support `None`, `bicubic`, `bilinear` or `nearest` interpolate mode")
        resize = transforms.Resize(
            size=(height, width),  # 目标尺寸 (height, width)
            interpolation=intermod  # 插值方法
        )
        logger.debug(f"Resize images to {(height, width)} by `{interstr}`")
    else:
        height, width = H, W
        resize = lambda x: x

    for i in range(N):
        img = origin_images[i] # (3, H, W)
        img = resize(img)
        img = torch.clamp(img, min=0, max=1.0)
        
        if mode == "pad":
            max_size = round(max(width, height) // 14) * 14
            h_padding = max_size - height
            w_padding = max_size - width
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
        else:  # mode == "crop"
            # Make divisible by 14
            new_width = (width // 14) * 14
            new_height = min((height // 14) * 14, new_width)

            start_y = (height - new_height) // 2
            img = img[:, start_y : start_y + new_height, :]
            start_x = (width - new_width) // 2
            img = img[:, :, start_x : start_x + new_width]
            
        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    return images