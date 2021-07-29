import os
import numpy as np
from PIL import Image


def show_mask_image(image_numpy):
    assert image_numpy.dtype == np.bool
    image_to_show = np.copy(image_numpy)
    image_to_show = (image_to_show * 255).astype(np.uint8)

    img = Image.fromarray(image_to_show)
    img.show()


def depth_image_to_grayscale(depth_image):
    return (depth_image * 255 / np.max(depth_image)).astype('uint8')


def overlay_images(image_original_1, image_original_2, alpha=0.5):
    from PIL import Image

    image_1 = np.copy(image_original_1)
    image_2 = np.copy(image_original_2)

    assert np.max(image_1) <= 1.0
    assert image_1.dtype == np.float32, image_1.dtype
    assert image_1.shape[0] == 3

    assert np.max(image_2) <= 1.0
    assert image_2.dtype == np.float32, image_2.dtype
    assert image_2.shape[0] == 3

    # Image 1
    image_1 = np.moveaxis(image_1, 0, -1) * 255.0
    image_1 = image_1.astype(np.uint8)

    image_1 = Image.fromarray(image_1)

    # Image 2
    image_2 = np.moveaxis(image_2, 0, -1) * 255.0
    image_2 = image_2.astype(np.uint8)

    image_2_alpha = np.ones((image_2.shape[0], image_2.shape[1], 1), dtype=np.float32) * 255.0 * alpha
    image_2_alpha = image_2_alpha.astype(np.uint8)

    image_2 = np.append(image_2, image_2_alpha, axis=2)
    image_2 = Image.fromarray(image_2)

    # Overlay
    image_1.paste(image_2, (0, 0), image_2)

    # Convert back to numpy
    image_1_np = np.array(image_1)

    return np.moveaxis(image_1_np, -1, 0).astype(np.float32) / 255.0


def overlay_mask(image_original, mask_original, alpha=0.5):
    from PIL import Image

    image = np.copy(image_original)
    mask = np.copy(mask_original)

    assert np.max(image) <= 1.0
    assert image.dtype == np.float32, image.dtype
    assert image.shape[0] == 3

    source = np.moveaxis(image, 0, -1) * 255.0
    source = source.astype(np.uint8)

    assert len(mask.shape) == 2
    assert np.max(mask) <= 1.0
    assert mask.dtype == np.bool

    mask = mask[:, :, np.newaxis] * 255.0
    mask = np.repeat(mask, 4, axis=-1)
    mask[..., -1] *= alpha
    mask = mask.astype(np.uint8)

    mask = Image.fromarray(mask)
    source = Image.fromarray(source)
    source.paste(mask, (0, 0), mask)

    return source


