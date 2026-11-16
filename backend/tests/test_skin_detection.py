"""Unit tests for supplementary skin presence detection across Fitzpatrick phototypes I-VI.

Verifies:
1. Detection across fair (I-II), olive/tan (III-IV), and dark/deeply pigmented (V-VI) phototypes.
2. Low-light / shadowed and bright dermatoscopic lighting conditions.
3. Accurate discrimination against non-skin objects (blue sky, green foliage, text documents).
4. Clinical lesion photos (pigmented lesion with surrounding perilesional skin).
5. Monitoring mode integrity: low skin fraction does not alter the acceptable decision.
"""

import numpy as np
import pytest
from PIL import Image

from src.services.quality_service import assess_image_quality, compute_skin_presence


def test_skin_detection_fitzpatrick_phototypes():
    """Verify skin detection across phototypes I through VI."""
    # Phototype I-II: Very fair / light skin with pinkish-peach tones
    fair_rgb = np.full((150, 150, 3), [242, 205, 185], dtype=np.uint8)
    detected, fraction = compute_skin_presence(fair_rgb)
    assert detected is True
    assert fraction > 0.90, f"Expected high fraction for fair skin, got {fraction}"

    # Phototype III-IV: Olive / medium / tan / South Asian cutaneous tone
    tan_rgb = np.full((150, 150, 3), [195, 145, 105], dtype=np.uint8)
    detected, fraction = compute_skin_presence(tan_rgb)
    assert detected is True
    assert fraction > 0.90, f"Expected high fraction for tan/olive skin, got {fraction}"

    # Phototype V-VI: Deep brown / dark African/Afro-Caribbean skin
    dark_rgb = np.full((150, 150, 3), [85, 55, 38], dtype=np.uint8)
    detected, fraction = compute_skin_presence(dark_rgb)
    assert detected is True
    assert fraction > 0.85, f"Expected high fraction for dark brown skin, got {fraction}"

    # Deepest phototype VI
    deep_rgb = np.full((150, 150, 3), [58, 38, 26], dtype=np.uint8)
    detected, fraction = compute_skin_presence(deep_rgb)
    assert detected is True
    assert fraction > 0.80, f"Expected high fraction for deep phototype VI, got {fraction}"


def test_skin_detection_varied_lighting_conditions():
    """Verify skin chromophore presence under dim, shadowed, and warm lighting."""
    # Dim/shadowed illumination (darkened fair-to-medium skin)
    dim_rgb = np.full((150, 150, 3), [120, 85, 65], dtype=np.uint8)
    detected, fraction = compute_skin_presence(dim_rgb)
    assert detected is True
    assert fraction > 0.85

    # Warm incandescent dermatoscopic illumination (red/amber shift)
    warm_rgb = np.full((150, 150, 3), [225, 150, 100], dtype=np.uint8)
    detected, fraction = compute_skin_presence(warm_rgb)
    assert detected is True
    assert fraction > 0.90


def test_skin_detection_non_skin_objects():
    """Verify rejection of non-cutaneous objects."""
    # 1. Blue sky / ocean
    blue_rgb = np.full((150, 150, 3), [50, 120, 220], dtype=np.uint8)
    detected, fraction = compute_skin_presence(blue_rgb)
    assert detected is False
    assert fraction == 0.0

    # 2. Green foliage / plant
    green_rgb = np.full((150, 150, 3), [35, 150, 45], dtype=np.uint8)
    detected, fraction = compute_skin_presence(green_rgb)
    assert detected is False
    assert fraction == 0.0

    # 3. White printed document
    white_rgb = np.full((150, 150, 3), [248, 248, 250], dtype=np.uint8)
    detected, fraction = compute_skin_presence(white_rgb)
    assert detected is False
    assert fraction == 0.0

    # 4. Neutral gray steel
    gray_rgb = np.full((150, 150, 3), [128, 128, 128], dtype=np.uint8)
    detected, fraction = compute_skin_presence(gray_rgb)
    assert detected is False
    assert fraction == 0.0


def test_skin_detection_lesion_photograph():
    """Verify skin detection on synthetic dermatoscopic lesion image."""
    # Create background skin (tan/olive, 200x200)
    img_arr = np.full((200, 200, 3), [195, 145, 110], dtype=np.uint8)

    # Add central hyperpigmented lesion (dark brown/black mole in center 80x80)
    img_arr[60:140, 60:140] = [40, 24, 18]

    # Add dark contact plate circular vignette border
    for y in range(200):
        for x in range(200):
            if (x - 100) ** 2 + (y - 100) ** 2 > 95 ** 2:
                img_arr[y, x] = [5, 5, 5]

    detected, fraction = compute_skin_presence(img_arr)
    assert detected is True
    assert fraction > 0.50, f"Expected lesion photograph skin fraction > 0.50, got {fraction}"


def test_monitoring_mode_preserves_acceptable_decision():
    """Confirm monitoring mode records skin presence without altering image acceptance decisions."""
    # Create a non-skin image with acceptable resolution, contrast, and sharpness
    rng = np.random.RandomState(42)
    non_skin_texture = rng.randint(50, 200, size=(200, 200, 3), dtype=np.uint8)
    # Ensure dominant blue/green channels so skin is not detected
    non_skin_texture[:, :, 0] = np.clip(non_skin_texture[:, :, 0] // 3, 0, 50)  # low red
    non_skin_texture[:, :, 2] = np.clip(non_skin_texture[:, :, 2] + 40, 100, 255) # high blue

    pil_img = Image.fromarray(non_skin_texture)
    quality = assess_image_quality(pil_img)

    # In monitoring mode:
    # 1. skin_detected should be False
    assert quality.skin_detected is False
    assert quality.skin_fraction < 0.05
    # 2. BUT acceptable MUST remain True because resolution, sharpness, and contrast are acceptable!
    assert quality.acceptable is True
    assert quality.reason is None
