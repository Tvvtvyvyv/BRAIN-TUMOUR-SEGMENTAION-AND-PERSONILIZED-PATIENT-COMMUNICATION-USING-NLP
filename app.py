import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras
from skimage.color import label2rgb
from skimage import morphology
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
from fpdf import FPDF
import io
from werkzeug.utils import secure_filename
from PIL import Image
import base64
from io import BytesIO

# Original dataclass definitions
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
        """Generate a natural language description of the MRI findings."""
        if "No tumor detected" in features or brain_lobe == "No tumor found":
            return ("The MRI scan appears to show healthy brain tissue with no evidence of tumors "
                   "or significant abnormalities. Continued routine monitoring is recommended.")
        
        size_desc = DescriptionGenerator._get_size_description(tumor_size)
        inflam_desc = DescriptionGenerator._get_inflammation_description(inflammation_intensity)
        
        description = f"The MRI scan reveals a {size_desc} tumor located in the {brain_lobe}. "
        
        if "Inflammation detected" in features:
            description += f"There is {inflam_desc} inflammation surrounding the tumor region. "
        
        if tumor_size > 0.20:
            description += ("Given the size of the tumor, close monitoring and prompt evaluation by a "
                          "specialist is recommended. ")
        else:
            description += ("Regular monitoring and follow-up imaging studies are recommended to track "
                          "any changes in the tumor size or characteristics. ")
        
        if "frontal" in brain_lobe.lower():
            description += ("The frontal lobe location may affect cognitive functions, behavior, and "
                          "motor skills. ")
        elif "parietal" in brain_lobe.lower():
            description += ("The parietal lobe location may impact sensory processing and spatial "
                          "awareness. ")
        elif "occipital" in brain_lobe.lower():
            description += ("The occipital lobe location may affect visual processing and "
                          "interpretation. ")
            
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

# Loss functions for the model
def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=4/3):
    epsilon = 1e-6
    y_true = keras.backend.flatten(y_true)
    y_pred = keras.backend.flatten(y_pred)
    
    true_pos = keras.backend.sum(y_true * y_pred)
    false_neg = keras.backend.sum(y_true * (1 - y_pred))
    false_pos = keras.backend.sum((1 - y_true) * y_pred)
    
    tversky = (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)
    return keras.backend.pow((1 - tversky), gamma)

def tversky(y_true, y_pred, alpha=0.7, beta=0.3):
    epsilon = 1e-6
    y_true = keras.backend.flatten(y_true)
    y_pred = keras.backend.flatten(y_pred)
    
    true_pos = keras.backend.sum(y_true * y_pred)
    false_neg = keras.backend.sum(y_true * (1 - y_pred))
    false_pos = keras.backend.sum((1 - y_true) * y_pred)
    
    return (true_pos + epsilon) / (true_pos + alpha * false_neg + beta * false_pos + epsilon)

# Analysis functions
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

def generate_description(mask: np.ndarray) -> Tuple[str, float, float, str]:
    """
    Generate a comprehensive description of MRI findings from the predicted mask.
    
    Args:
        mask: Binary prediction mask from the model
    
    Returns:
        Tuple containing:
        - description (str): Natural language description of findings
        - tumor_size (float): Relative size of tumor
        - inflammation_intensity (float): Measure of inflammation (0-1)
        - brain_lobe (str): Affected brain region
    """
    # Calculate tumor size as percentage of brain area
    tumor_size = np.sum(mask > 0.5) / mask.size
    
    # Calculate inflammation intensity
    inflammation_mask = detect_inflammation(mask)
    inflammation_intensity = np.mean(mask[inflammation_mask]) if np.any(inflammation_mask) else 0.0
    
    # Get features description
    features = describe_features(mask)
    
    # Determine affected brain lobe
    brain_lobe = determine_brain_lobe(mask)
    
    # Generate natural language description
    description = DescriptionGenerator.generate_description(
        features=features,
        tumor_size=tumor_size,
        inflammation_intensity=inflammation_intensity,
        brain_lobe=brain_lobe
    )
    
    return description, tumor_size, inflammation_intensity, brain_lobe

# PDF Report Generation
class MedicalReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Brain MRI Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def patient_info(self, name, age, date):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Patient Information:', 0, 1)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f'Name: {name}', 0, 1)
        self.cell(0, 10, f'Age: {age}', 0, 1)
        self.cell(0, 10, f'Date: {date}', 0, 1)
        self.ln(10)

    def add_findings(self, results):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Clinical Findings:', 0, 1)
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, results['description'])
        self.ln(10)

        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Quantitative Measurements:', 0, 1)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, f"Tumor Size: {results['tumor_size']:.1%} of visible brain area", 0, 1)
        self.cell(0, 10, f"Inflammation Intensity: {results['inflammation_intensity']:.2f} (0-1 scale)", 0, 1)
        self.cell(0, 10, f"Affected Brain Region: {results['brain_lobe']}", 0, 1)

def analyze_mri_scan(image_path, model_path='/Users/sandeepkumar/Documents/Image project/unet_brain_mri_seg.keras'):
    """Analyze a single MRI scan and return the results."""
    model = keras.models.load_model(model_path, 
                                  custom_objects={'focal_tversky': focal_tversky, 'tversky': tversky})
    
    x = np.asarray(load_img(image_path, target_size=(256, 256)))
    x = np.expand_dims(x, axis=0)
    
    pred = model.predict(x / 255)
    pred_t = (pred > 0.5).astype(np.uint8)
    
    # Generate overlay image
    overlay = label2rgb(pred_t[0].squeeze(), image=x[0]/255, bg_label=0, colors=[(0, 1, 0)], alpha=0.3)
    
    # Save overlay to bytes for web display
    img_bytes = BytesIO()
    plt.imsave(img_bytes, overlay)
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
    
    # Generate description and measurements
    description, tumor_size, inflammation_intensity, brain_lobe = generate_description(pred_t[0])
    feature_description = describe_features(pred_t[0])
    
    return {
        'description': description,
        'features': feature_description,
        'tumor_size': tumor_size,
        'inflammation_intensity': inflammation_intensity,
        'brain_lobe': brain_lobe,
        'image_data': f'data:image/png;base64,{img_base64}'
    }

# Flask Application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    patient_name = request.form.get('patient_name', 'Unknown')
    patient_age = request.form.get('patient_age', 'Unknown')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = analyze_mri_scan(filepath)
            results['patient_name'] = patient_name
            results['patient_age'] = patient_age
            os.remove(filepath)
            
            return jsonify(results)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    
    pdf = MedicalReport()
    pdf.add_page()
    pdf.patient_info(
        data['patient_name'],
        data['patient_age'],
        datetime.now().strftime("%Y-%m-%d")
    )
    pdf.add_findings(data)
    
    pdf_content = pdf.output(dest='S').encode('latin1')
    
    return send_file(
        io.BytesIO(pdf_content),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"mri_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
if __name__ == '__main__':
    app.run(debug=True)