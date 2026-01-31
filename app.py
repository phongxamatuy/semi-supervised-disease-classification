
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS 
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATH = 'models/05_semi_supervised_model.pth'

diseases_en = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

diseases_vi = {
    'Atelectasis': 'Xẹp phổi',
    'Consolidation': 'Đông đặc phổi',
    'Infiltration': 'Thâm nhiễm phổi',
    'Pneumothorax': 'Tràn khí màng phổi',
    'Edema': 'Phù phổi',
    'Emphysema': 'Khí phế thũng',
    'Fibrosis': 'Xơ hóa phổi',
    'Effusion': 'Tràn dịch màng phổi',
    'Pneumonia': 'Viêm phổi',
    'Pleural_Thickening': 'Dày màng phổi',
    'Cardiomegaly': 'Tim to',
    'Nodule': 'Nốt',
    'Mass': 'Khối u',
    'Hernia': 'Thoát vị'
}



def load_model():
    """Load mô hình"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 14)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key.replace('module.', '', 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ Model loaded successfully!")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")
        return None
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()


def preprocess_image(image):
    """Tiền xử lý ảnh X-quang"""
    try:
   
        if image.mode != 'L':
            image = image.convert('L')
        
       
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
      
        image_array = np.array(image).astype(np.float32) / 255.0
        
     
        image_3ch = np.stack([image_array] * 3, axis=-1)
        
       
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_normalized = (image_3ch - mean) / std
        
     
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(DEVICE)
    except Exception as e:
        print(f"❌ Error in preprocess_image: {e}")
        return None

def predict(image):
    """Dự đoán bệnh"""
    if model is None:
        print("⚠️ Model is None")
        return None
    
    print(f"🔄 Predicting... Image type: {type(image)}")
    
    image_tensor = preprocess_image(image)
    if image_tensor is None:
        print("⚠️ Image preprocessing failed")
        return None
    
    print(f"✅ Image tensor shape: {image_tensor.shape}")
    
    try:
        with torch.no_grad():
            output = model(image_tensor)
            print(f"✅ Model output shape: {output.shape}")
            probs = torch.sigmoid(output).cpu().numpy()[0]
            print(f"✅ Probabilities extracted successfully")
        
      
        results = []
        for i, disease_en in enumerate(diseases_en):
            results.append({
                'disease_vi': diseases_vi.get(disease_en, disease_en),
                'disease_en': disease_en,
                'probability': float(probs[i]),
                'percentage': float(probs[i] * 100),
            })
        
      
        results.sort(key=lambda x: x['probability'], reverse=True)
        print(f"✅ Prediction successful! Total results: {len(results)}")
        return results
    
    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST', 'OPTIONS']) 
def api_predict():
    """API dự đoán"""
    
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
       
        image = Image.open(io.BytesIO(file.read()))
        print(f"✅ Image loaded: {image.size}, mode: {image.mode}")
        
   
        results = predict(image)
        
        if results is None:
            return jsonify({'error': 'Error predicting'}), 500
        
       
        detected = [r for r in results if r['probability'] >= 0.5]
        
        response = {
            'status': 'success',
            'detected_diseases': detected,
            'all_predictions': results,
            'num_detected': len(detected)
        }
        
        print(f"✅ Response: {len(detected)} diseases detected")
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in api_predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Check health"""
    return jsonify({
        'status': 'healthy',
        'model': 'loaded' if model is not None else 'not_loaded',
        'device': str(DEVICE)
    })

if __name__ == '__main__':
    print("="*80)
    print("🏥 CHEST X-RAY DISEASE CLASSIFIER")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {'✅ Ready' if model is not None else '❌ Not loaded'}")
    print("\n🚀 Server running at http://localhost:5000")
    print("⚠️  Make sure to access from http://localhost:5000, NOT Live Server!")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)