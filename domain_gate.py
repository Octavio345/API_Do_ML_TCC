import numpy as np
import cv2


MIN_VEGETATION_RATIO = 0.08
MIN_VEGETATION_COMPONENT_RATIO = 0.04
MIN_EDGE_DENSITY_IN_VEGETATION = 0.08
MIN_HUE_STD_IN_VEGETATION = 6.0
SECONDARY_SCORE_THRESHOLD = 0.45
SECONDARY_WEIGHTS = {
    "edge_density_in_vegetation": 0.6,
    "hue_std_in_vegetation": 0.4,
}

MIN_SHARPNESS = 18.0


def analyze_quality_and_domain(rgb_image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)

    veg_mask = (
        (hue >= 32) & (hue <= 88)
        & (saturation >= 55)
        & (value >= 40)
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    veg_mask_clean = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)
    veg_mask_clean = cv2.morphologyEx(veg_mask_clean, cv2.MORPH_CLOSE, kernel)

    vegetation_ratio = float(np.mean(veg_mask_clean))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        veg_mask_clean, connectivity=8
    )
    if num_labels > 1:
        largest_component_ratio = float(stats[1:, cv2.CC_STAT_AREA].max()) / veg_mask_clean.size
    else:
        largest_component_ratio = 0.0

    veg_pixel_mask = veg_mask_clean > 0
    veg_pixel_count = int(np.sum(veg_pixel_mask))

    veg_pixels_hue = hue[veg_pixel_mask]
    hue_std_in_vegetation = float(np.std(veg_pixels_hue)) if veg_pixel_count > 0 else 0.0

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if veg_pixel_count > 0:
        edge_pixels_in_vegetation = int(np.sum((edges > 0) & veg_pixel_mask))
        edge_density_in_vegetation = float(edge_pixels_in_vegetation) / float(veg_pixel_count)
    else:
        edge_density_in_vegetation = 0.0

    return {
        "vegetation_ratio": round(vegetation_ratio, 4),
        "vegetation_component_ratio": round(largest_component_ratio, 4),
        "edge_density_in_vegetation": round(edge_density_in_vegetation, 4),
        "hue_std_in_vegetation": round(hue_std_in_vegetation, 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(np.mean(gray)), 2),
    }


def secondary_texture_score(quality: dict) -> float:
    normalized = {
        "edge_density_in_vegetation": quality["edge_density_in_vegetation"] / MIN_EDGE_DENSITY_IN_VEGETATION,
        "hue_std_in_vegetation": quality["hue_std_in_vegetation"] / MIN_HUE_STD_IN_VEGETATION,
    }

    score = 0.0
    for key, weight in SECONDARY_WEIGHTS.items():
        score += weight * min(normalized[key], 1.0)

    return round(score, 4)


def is_out_of_domain(quality: dict) -> tuple[bool, str | None]:
    has_enough_vegetation = quality["vegetation_ratio"] >= MIN_VEGETATION_RATIO
    has_coherent_blob = quality["vegetation_component_ratio"] >= MIN_VEGETATION_COMPONENT_RATIO

    if not (has_enough_vegetation and has_coherent_blob):
        return True, "A imagem nao parece conter vegetacao/lavoura suficiente para diagnostico de soja."

    if secondary_texture_score(quality) < SECONDARY_SCORE_THRESHOLD:
        return True, "A area verde detectada nao apresenta textura compativel com folha de soja."

    return False, None


def is_low_quality(quality: dict) -> tuple[bool, str | None]:
    if quality["sharpness"] < MIN_SHARPNESS:
        return True, "Imagem com pouca nitidez. Envie uma foto mais clara da planta/lavoura."
    return False, None