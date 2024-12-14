import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter

def MinFilter(image, size=3):
    """
    Apply min filter to an image
    Args:
        image: Input image
        size: Size of the filter kernel (default: 3)
    """
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply min filter to each channel if RGB
    if len(image.shape) == 3:
        filtered = np.zeros_like(image)
        for channel in range(image.shape[2]):
            filtered[:,:,channel] = minimum_filter(image[:,:,channel], size=size)
    else:
        filtered = minimum_filter(image, size=size)
    
    return filtered.astype(np.uint8)