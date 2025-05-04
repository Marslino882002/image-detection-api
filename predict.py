#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, fields
import torch
import cv2
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
import io
import os

# Initialize Flask app first (before any ML imports)
app = Flask(__name__)
api = Api(app, version="1.0", title="Food Weight Estimation API", 
          description="API for estimating food weight using YOLOv8")

# Configuration (defined before ML imports)
TARGET_CLASSES = {
    46: {'name': 'banana', 'range': (100, 120), 'shape': 'cylinder', 'ref_width': 3.5},
    47: {'name': 'apple', 'range': (70, 100), 'shape': 'sphere', 'ref_width': 7.0},
    48: {'name': 'sandwich', 'range': (150, 400), 'shape': 'cuboid', 'ref_width': 12.0},
    49: {'name': 'orange', 'range': (100, 131), 'shape': 'sphere', 'ref_width': 8.0},
    51: {'name': 'carrot', 'range': (50, 100), 'shape': 'cylinder', 'ref_width': 2.5},
    53: {'name': 'pizza', 'range': (200, 400), 'shape': 'disk', 'ref_width': 30.0}
}

# Now import ML components
try:
    from ultralytics import YOLO
    model = YOLO("yolo_weight_estimation22.pt")
    print("✅ YOLO model loaded successfully")
except ImportError as e:
    print(f"❌ Error importing YOLO: {e}")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

ns = api.namespace("predict", description="Food weight prediction operations")

# Request parser
upload_parser = api.parser()
upload_parser.add_argument('file', type=FileStorage, location='files', required=True, 
                         help='JPEG, JPG or PNG image file')

# Response model
weight_model = api.model('WeightEstimation', {
    'object': fields.String(required=True, description='Detected food item'),
    'weight_grams': fields.Float(required=True, description='Estimated weight in grams'),
    'confidence': fields.Float(required=True, description='Detection confidence'),
    'bounding_box': fields.List(fields.Float, description='Bounding box coordinates [x1,y1,x2,y2]')
})

def calculate_weight(cls_id, area_ratio):
    """Calculate normalized weight based on size ratio"""
    cls_info = TARGET_CLASSES.get(cls_id)
    if not cls_info:
        return 0
    min_weight, max_weight = cls_info['range']
    return min_weight + (max_weight - min_weight) * area_ratio

@ns.route('/')
class WeightEstimation(Resource):
    @api.expect(upload_parser)
    @api.marshal_list_with(weight_model)
    @api.response(400, 'Invalid file format')
    @api.response(500, 'Model not loaded or processing error')
    def post(self):
        """Estimate food weights from an uploaded image"""
        if model is None:
            api.abort(500, "Model not loaded")
            
        args = upload_parser.parse_args()
        file = args['file']
        
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            api.abort(400, "Only JPG/JPEG/PNG files are allowed")
            
        try:
            # Read and convert image
            img_bytes = file.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_height, img_width = img.shape[:2]
            img_area = img_height * img_width
            
            # Run detection
            results = model(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            
            # Process results
            predictions = []
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in TARGET_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bbox_area = (x2 - x1) * (y2 - y1)
                        area_ratio = bbox_area / img_area
                        
                        predictions.append({
                            'object': TARGET_CLASSES[cls_id]['name'],
                            'weight_grams': round(calculate_weight(cls_id, area_ratio), 1),
                            'confidence': round(float(box.conf[0]), 2),
                            'bounding_box': [x1, y1, x2, y2]
                        })
            
            return predictions
            
        except Exception as e:
            api.abort(500, f"Processing error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


# In[1]:


import torch
print(torch.__version__)  # Should work without error


# In[19]:


get_ipython().system('pip install torch --upgrade')


# In[ ]:




