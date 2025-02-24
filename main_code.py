import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
from skimage.color import label2rgb
from skimage import morphology
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class LobeRegion:
    name: str
    y_range: Tuple[float, float]
    color: Tuple[int, int, int]

class DescriptionGenerator:
    """Generates natural language descriptions of brain MRI findings."""
    
    @staticmethod
    def _get_size_description(tumor_size: float) -> str:
        if tumor_size < 0.05:
            return "very small"
        elif tumor_size < 0.10:
            return "small"
        elif tumor_size < 0.20:
            return "moderate-sized"
        elif tumor_size < 0.30:
            return "large"
        else:
            return "very large"
    
    @staticmethod
    def _get_inflammation_description(intensity: float) -> str:
        if intensity < 0.3:
            return "minimal"
        elif intensity < 0.5:
            return "moderate"
        elif intensity < 0.7:
            return "significant"
        else:
            return "severe"
    
    @staticmethod
    def generate_description(features: List[str], tumor_size: float, 
                           inflammation_intensity: float, brain_lobe: str) -> str:
        """
        Generate a natural language description of the MRI findings.
        """
        if "No tumor detected" in features or brain_lobe == "No tumor found":
            return ("The MRI scan appears to show healthy brain tissue with no evidence of tumors "
                   "or significant abnormalities. Continued routine monitoring is recommended.")
        
        # Build description based on findings
        size_desc = DescriptionGenerator._get_size_description(tumor_size)
        inflam_desc = DescriptionGenerator._get_inflammation_description(inflammation_intensity)
        
        description = (
            f"The MRI scan reveals a {size_desc} tumor located in the {brain_lobe}. "
        )
        
        if "Inflammation detected" in features:
            description += (
                f"There is {inflam_desc} inflammation surrounding the tumor region. "
            )
        
        # Add clinical implications
        if tumor_size > 0.20:
            description += (
                "Given the size of the tumor, close monitoring and prompt evaluation by a "
                "specialist is recommended. "
            )
        else:
            description += (
                "Regular monitoring and follow-up imaging studies are recommended to track "
                "any changes in the tumor size or characteristics. "
            )
        
        # Add anatomical context
        if "frontal" in brain_lobe.lower():
            description += (
                "The frontal lobe location may affect cognitive functions, behavior, and "
                "motor skills. "
            )
        elif "parietal" in brain_lobe.lower():
            description += (
                "The parietal lobe location may impact sensory processing and spatial "
                "awareness. "
            )
        elif "occipital" in brain_lobe.lower():
            description += (
                "The occipital lobe location may affect visual processing and "
                "interpretation. "
            )
            
        return description

class BrainLobeDetector:
    def __init__(self):
        self.regions = [
            LobeRegion("frontal", (0.0, 0.4), (255, 255, 200)),
            LobeRegion("parietal", (0.4, 0.7), (200, 200, 200)),
            LobeRegion("occipital", (0.7, 1.0), (200, 255, 200)),
        ]
        self.longitudinal_fissure_x = 0.5
    
    def determine_hemisphere(self, x_coord: float, width: int) -> str:
        normalized_x = x_coord / width
        return 'left' if normalized_x < self.longitudinal_fissure_x else 'right'
    
    def detect_lobe(self, tumor_mask: np.ndarray) -> Tuple[str, float, List[str]]:
        if not np.any(tumor_mask > 0.5):
            return "No tumor detected", 0.0, ["No tumor found"]
            
        y_coords, x_coords = np.where(tumor_mask > 0.5)
        if len(y_coords) == 0:
            return "No tumor detected", 0.0, ["No tumor found"]
            
        center_y = np.mean(y_coords) / tumor_mask.shape[0]
        center_x = np.mean(x_coords) / tumor_mask.shape[1]
        
        hemisphere = self.determine_hemisphere(np.mean(x_coords), tumor_mask.shape[1])
        
        primary_lobe = None
        max_confidence = 0.0
        affected_regions = []
        
        for region in self.regions:
            if region.y_range[0] <= center_y <= region.y_range[1]:
                relative_position = (center_y - region.y_range[0]) / (region.y_range[1] - region.y_range[0])
                confidence = 1.0 - 2 * abs(0.5 - relative_position)
                confidence = max(0.0, min(1.0, confidence))
                
                affected_regions.append(f"{hemisphere} {region.name}")
                if confidence > max_confidence:
                    max_confidence = confidence
                    primary_lobe = f"{hemisphere} {region.name}"
        
        tumor_extent = (np.max(y_coords) - np.min(y_coords)) / tumor_mask.shape[0]
        if tumor_extent > 0.3:
            return f"Large tumor spanning multiple regions: {', '.join(affected_regions)}", max_confidence, affected_regions
            
        return primary_lobe, max_confidence, affected_regions

def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3):
    """
    Focal Tversky loss function for image segmentation.
    """
    epsilon = 1e-6
    y_true = keras.backend.flatten(y_true)
    y_pred = keras.backend.flatten(y_pred)
    
    true_pos = keras.backend.sum(y_true * y_pred)
    false_neg = keras.backend.sum(y_true * (1 - y_pred))
    false_pos = keras.backend.sum((1 - y_true) * y_pred)
    
    tversky = (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    return keras.backend.pow((1 - tversky), gamma)

def tversky(y_true, y_pred, alpha=0.7, beta=0.3):
    """
    Tversky loss function for image segmentation.
    """
    epsilon = 1e-6
    y_true = keras.backend.flatten(y_true)
    y_pred = keras.backend.flatten(y_pred)
    
    true_pos = keras.backend.sum(y_true * y_pred)
    false_neg = keras.backend.sum(y_true * (1 - y_pred))
    false_pos = keras.backend.sum((1 - y_true) * y_pred)
    
    return (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)

def load_and_process_images(image_paths, mask_paths, target_size=(256, 256)):
    num_images = len(image_paths)
    
    x = np.zeros((num_images, target_size[0], target_size[1], 3), dtype="float32")
    disp_x = np.zeros((num_images, target_size[0], target_size[1], 3), dtype="uint8")
    y = np.zeros((num_images, target_size[0], target_size[1], 1), dtype="uint8")
    
    for j in range(num_images):
        x[j] = np.asarray(load_img(image_paths[j], target_size=target_size))
        disp_x[j] = x[j]
        
        mask_img = np.asarray(load_img(mask_paths[j], target_size=target_size, color_mode="grayscale"))
        y[j] = np.expand_dims(mask_img, 2)
    
    return x, disp_x, y

def detect_inflammation(mask, dynamic_threshold=True):
    if dynamic_threshold:
        threshold = np.percentile(mask, 95)
    else:
        threshold = 0.3
    
    inflammation_regions = mask >= threshold
    cleaned_inflammation = morphology.remove_small_objects(inflammation_regions, min_size=50)
    return cleaned_inflammation

def describe_features(predicted_mask):
    description = []
    
    if np.any(predicted_mask > 0.5):
        description.append("Tumor region detected")
        cleaned_inflammation = detect_inflammation(predicted_mask)
        if np.any(cleaned_inflammation):
            description.append("Inflammation detected")
    else:
        description.append("No tumor detected")
        description.append("No inflammation found")
        description.append("Healthy tissue observed")
    
    return description

def determine_brain_lobe(mask: np.ndarray) -> str:
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim > 2:
        mask = mask[:, :, 0]
        
    detector = BrainLobeDetector()
    binary_mask = (mask > 0.5).astype(np.uint8)
    primary_lobe, confidence, affected_regions = detector.detect_lobe(binary_mask)
    
    if primary_lobe == "No tumor detected":
        return "No tumor found"
    
    if confidence < 0.3:
        return f"Tumor detected in border region between {', '.join(affected_regions)}"
    
    return f"Tumor primarily in {primary_lobe} lobe ({confidence:.1%} confidence)"

def generate_description(feature_description, predicted_mask):
    if not np.any(predicted_mask > 0.5):
        return (
            "The MRI scan appears to show healthy brain tissue with no evidence of tumors "
            "or significant abnormalities. Continued routine monitoring is recommended.",
            0.0,
            0.0,
            "No tumor found"
        )
    
    tumor_size = np.sum(predicted_mask > 0.5) / (predicted_mask.shape[0] * predicted_mask.shape[1])
    inflammation_intensity = np.mean(predicted_mask[predicted_mask > 0.5]) if np.any(predicted_mask > 0.5) else 0
    brain_lobe = determine_brain_lobe(predicted_mask)
    
    description = DescriptionGenerator.generate_description(
        feature_description,
        tumor_size,
        inflammation_intensity,
        brain_lobe
    )
    
    return description, tumor_size, inflammation_intensity, brain_lobe

def visualize_images_with_features(x, predictions, feature_descriptions, descriptions):
    num_images = len(x)
    
    plt.figure(figsize=(12, num_images * 10))
    
    for j in range(num_images):
        img = x[j] / 255.0
        mask = predictions[j].squeeze()
        
        overlay = label2rgb(mask, image=img, bg_label=0, colors=[(0, 1, 0)], alpha=0.3)
        
        ax = plt.subplot(num_images, 1, j + 1)
        
        plt.imshow(overlay)
        plt.axis("off")
        
        title = "\n".join(feature_descriptions[j])
        ax.set_title(title, fontsize=16, color='red' if "Tumor region detected" in title else 'green')
        
        plt.text(0, -0.1, descriptions[j], fontsize=12, wrap=True, 
                transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def analyze_mri_scan(image_path, model_path='unet_brain_mri_seg.keras'):
    """
    Analyze a single MRI scan and return the results.
    """
    # Load model
    model = keras.models.load_model(model_path, 
                                  custom_objects={'focal_tversky': focal_tversky, 'tversky': tversky})
    
    # Process single image
    x = np.asarray(load_img(image_path, target_size=(256, 256)))
    x = np.expand_dims(x, axis=0)
    disp_x = x.copy()
    
    # Make prediction
    pred = model.predict(x / 255)
    pred_t = (pred > 0.5).astype(np.uint8)
    
    # Generate analysis
    feature_description = describe_features(pred_t[0])
    description, tumor_size, inflammation_intensity, brain_lobe = generate_description(
        feature_description, pred_t[0]
    )
    
    # Visualize
    visualize_images_with_features(disp_x, pred_t, [feature_description], [description])
    
    return {
        'description': description,
        'features': feature_description,
        'tumor_size': tumor_size,
        'inflammation_intensity': inflammation_intensity,
        'brain_lobe': brain_lobe
    }

if __name__ == "__main__":
    # Example usage
    image_path = '/Users/sandeepkumar/Downloads/archive/lgg-mri-segmentation/kaggle_3m/TCGA_HT_8114_19981030/TCGA_HT_8114_19981030_19.tif'
    results = analyze_mri_scan(image_path)
    
    print("\nMRI Analysis Results:")
    print("-" * 50)
    print("\nDetailed Description:")
    print(results['description'])
    print("\nDetected Features:")
    for feature in results['features']:
        print(f"- {feature}")
    print("\nQuantitative Measurements:")
    print(f"- Tumor size: {results['tumor_size']:.1%} of visible brain area")
    print(f"- Inflammation intensity: {results['inflammation_intensity']:.2f} (0-1 scale)")
    print(f"- Affected brain region: {results['brain_lobe']}")