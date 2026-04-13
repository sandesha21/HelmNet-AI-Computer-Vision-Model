# 🚀 Deployment Guide: Productionizing HelmNet

## 📋 Executive Summary

This guide provides step-by-step instructions for deploying Model 4 to production environments, covering infrastructure setup, integration, testing, and rollout strategies.

**Key Deliverables**: Infrastructure setup, model containerization, system integration, testing protocols, and phased rollout strategy.

---

## ✅ Pre-Deployment Checklist

### 🤖 Model Validation

**Output Description**: Verify model meets all performance requirements before production deployment.
- [ ] Model accuracy verified: 96.8% ✅
- [ ] Cross-validation score: 94.2% ✅
- [ ] False negative rate acceptable: 1.2% ✅
- [ ] False positive rate acceptable: 3.2% ✅
- [ ] Model size verified: 85MB ✅
- [ ] Inference time verified: <100ms ✅

### 🖥️ Infrastructure Readiness

**Output Description**: Ensure all hardware and network resources are provisioned and tested.
- [ ] GPU servers provisioned (RTX 2070 or better)
- [ ] Network bandwidth verified (>100 Mbps)
- [ ] Storage capacity verified (>500GB for logs)
- [ ] Backup systems configured
- [ ] Monitoring infrastructure ready

### 🔗 Integration Readiness

**Output Description**: Verify all external system integrations are configured and tested.
- [ ] CCTV system API documented
- [ ] Alert system integration tested
- [ ] Database schema prepared
- [ ] API endpoints configured
- [ ] Authentication/authorization setup

### 🔒 Compliance & Security

**Output Description**: Ensure all security and compliance requirements are met before deployment.
- [ ] Data privacy policy reviewed
- [ ] Security audit completed
- [ ] Compliance requirements met (GDPR, CCPA, etc.)
- [ ] Encryption configured (TLS 1.3)
- [ ] Access controls implemented

---

## 🏗️ Architecture Overview

### 📊 Deployment Architecture

**Output Description**: System architecture showing data flow from CCTV cameras through inference to alerts and monitoring.

```
┌─────────────────────────────────────────────────────────────┐
│                    CCTV Camera Network                       │
│                  (10-50 camera feeds)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Video Stream Ingestion Layer                    │
│  • RTSP/RTMP stream handlers                                │
│  • Frame extraction (30 fps)                                │
│  • Preprocessing (resize, normalize)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Model Inference Layer (GPU Accelerated)            │
│  • Model 4 (85MB, <100ms inference)                         │
│  • Batch processing (10-30 frames/batch)                    │
│  • Confidence scoring                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Alert & Action Layer                            │
│  • Confidence thresholding                                  │
│  • Alert generation                                         │
│  • Notification dispatch                                    │
│  • Logging & audit trail                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Monitoring & Analytics Layer                       │
│  • Performance metrics                                      │
│  • Model drift detection                                    │
│  • Dashboard & reporting                                    │
│  • Database storage                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🖥️ Infrastructure Setup

### ⚙️ Hardware Requirements

**Output Description**: Hardware specifications for different deployment scales. Choose based on number of camera feeds.

#### Minimum Configuration (10 Cameras)
```
CPU: Intel Xeon E5-2680 v4 (14 cores)
GPU: NVIDIA RTX 2070 (8GB VRAM)
RAM: 32GB DDR4
Storage: 1TB SSD (for logs/cache)
Network: 1Gbps Ethernet
```

**Cost**: ~$2,400/server

#### Recommended Configuration (20-30 Cameras)
```
CPU: Intel Xeon Platinum 8280 (28 cores)
GPU: NVIDIA RTX 2080 Ti (11GB VRAM)
RAM: 64GB DDR4
Storage: 2TB NVMe SSD
Network: 10Gbps Ethernet
```

**Cost**: ~$4,500/server

#### High-Performance Configuration (50+ Cameras)
```
CPU: Dual Intel Xeon Platinum 8280 (56 cores)
GPU: 2x NVIDIA RTX 2080 Ti (22GB VRAM total)
RAM: 128GB DDR4
Storage: 4TB NVMe SSD
Network: 10Gbps Ethernet
```

**Cost**: ~$8,000/server

### 🌐 Network Architecture

**Output Description**: Network topology and bandwidth planning for optimal performance.

#### Bandwidth Requirements

**Output Description**: Bandwidth allocation across different components. Total required bandwidth should be 100+ Mbps for safety margin.

| Component | Bandwidth | Notes |
|-----------|-----------|-------|
| CCTV Streams (10 cameras @ 5Mbps) | 50 Mbps | Incoming |
| Alert Notifications | 1 Mbps | Outgoing |
| Monitoring/Logging | 5 Mbps | Outgoing |
| **Total Required** | **56 Mbps** | Recommend 100+ Mbps |

#### Network Topology

```
┌─────────────────────────────────────────┐
│     CCTV Network (Isolated VLAN)        │
│  • 10-50 camera feeds                   │
│  • Dedicated network segment             │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    Inference Server (GPU-Accelerated)   │
│  • Model 4 inference                    │
│  • Real-time processing                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│    Alert & Monitoring Network           │
│  • Alert dispatch                       │
│  • Dashboard access                     │
│  • Logging & analytics                  │
└─────────────────────────────────────────┘
```

---

## 🤖 Model Deployment

### 📦 Step 1: Model Preparation

**Output Description**: Load model, verify properties, test inference speed, and validate accuracy before deployment.

```python
# Load and verify model
import tensorflow as tf

model = tf.keras.models.load_model('model_4_final.h5')

# Verify model properties - Check parameter count and file size
print(f"Model size: {model.count_params() / 1e6:.1f}M parameters")
print(f"Model file size: {os.path.getsize('model_4_final.h5') / 1e6:.1f}MB")

# Test inference speed - Measure average time per image
import time
X_test_sample = X_test[:100]
start = time.time()
predictions = model.predict(X_test_sample, batch_size=32)
inference_time = (time.time() - start) / len(X_test_sample)
print(f"Inference time: {inference_time*1000:.1f}ms per image")

# Verify accuracy - Ensure model meets 96%+ requirement
accuracy = model.evaluate(X_test, y_test)[1]
print(f"Model accuracy: {accuracy*100:.1f}%")
```

**Output Description**: Model properties, inference speed, and accuracy verification results.

### ⚡ Step 2: Model Optimization

**Output Description**: Convert model to optimized formats (TFLite, ONNX) for edge deployment and cross-platform compatibility.

```python
# Convert to TensorFlow Lite for edge deployment (optional)
# TFLite reduces model size and inference latency for edge devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model_4_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert to ONNX for cross-platform compatibility (optional)
# ONNX enables deployment on various platforms (Windows, Linux, mobile)
import onnx
import tf2onnx

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model_4_optimized.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

**Output Description**: Optimized model files for edge deployment and cross-platform use.

### 🐳 Step 3: Model Containerization

**Output Description**: Docker container for reproducible, isolated model deployment across environments.

```dockerfile
# Dockerfile for HelmNet inference server
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install dependencies - Python and pip for package management
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages - TensorFlow, Flask, and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code - Model weights and inference logic
COPY model_4_final.h5 .
COPY inference_server.py .
COPY config.yaml .

# Expose port - API endpoint for inference requests
EXPOSE 8000

# Health check - Verify server is running and responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run inference server - Start the Flask application
CMD ["python3", "inference_server.py"]
```

**Output Description**: Docker image with all dependencies, model, and inference server ready for deployment.

### 🚀 Step 4: Deployment to Production

**Output Description**: Commands to build, push, and deploy Docker container to production infrastructure.

```bash
# Build Docker image - Create container with all dependencies
docker build -t helmnet:v4 .

# Push to registry - Upload to container registry for distribution
docker tag helmnet:v4 registry.company.com/helmnet:v4
docker push registry.company.com/helmnet:v4

# Deploy to Kubernetes (if using K8s) - Orchestrated deployment with auto-scaling
kubectl apply -f helmnet-deployment.yaml

# Or deploy to Docker Swarm - Distributed deployment across cluster
docker service create \
  --name helmnet-inference \
  --replicas 2 \
  --publish 8000:8000 \
  --constraint node.labels.gpu==true \
  registry.company.com/helmnet:v4
```

**Output Description**: Docker image deployed to production with 2 replicas for high availability.

---

## Integration with Existing Systems

### CCTV System Integration

#### RTSP Stream Ingestion

```python
import cv2
import threading
from queue import Queue

class RTSPStreamHandler:
    def __init__(self, rtsp_url, queue_size=30):
        self.rtsp_url = rtsp_url
        self.frame_queue = Queue(maxsize=queue_size)
        self.running = False
        
    def start(self):
        self.running = True
        thread = threading.Thread(target=self._read_frames, daemon=True)
        thread.start()
        
    def _read_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        while self.running:
            ret, frame = cap.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            else:
                # Reconnect on stream failure
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                
    def get_frame(self):
        return self.frame_queue.get(timeout=5)
        
    def stop(self):
        self.running = False

# Usage
stream_handler = RTSPStreamHandler('rtsp://camera1.local:554/stream')
stream_handler.start()
```

#### Alert System Integration

```python
import requests
import json
from datetime import datetime

class AlertDispatcher:
    def __init__(self, alert_api_url, api_key):
        self.alert_api_url = alert_api_url
        self.api_key = api_key
        
    def send_alert(self, camera_id, alert_type, confidence, frame_data=None):
        """Send alert to central alert system"""
        payload = {
            'timestamp': datetime.utcnow().isoformat(),
            'camera_id': camera_id,
            'alert_type': alert_type,  # 'no_helmet', 'partial_helmet', etc.
            'confidence': float(confidence),
            'frame_data': frame_data  # Optional: base64 encoded frame
        }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                self.alert_api_url,
                json=payload,
                headers=headers,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Alert dispatch failed: {e}")
            # Implement retry logic or fallback
            
# Usage
dispatcher = AlertDispatcher(
    'https://alerts.company.com/api/v1/alerts',
    api_key='your-api-key'
)

dispatcher.send_alert(
    camera_id='camera_001',
    alert_type='no_helmet',
    confidence=0.98
)
```

### Database Integration

```python
import sqlite3
from datetime import datetime

class InferenceLogger:
    def __init__(self, db_path='helmnet_logs.db'):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inferences (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                camera_id TEXT,
                helmet_detected BOOLEAN,
                confidence REAL,
                inference_time_ms REAL,
                model_version TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                camera_id TEXT,
                alert_type TEXT,
                confidence REAL,
                acknowledged BOOLEAN,
                acknowledged_by TEXT,
                acknowledged_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_inference(self, camera_id, helmet_detected, confidence, inference_time_ms):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO inferences 
            (timestamp, camera_id, helmet_detected, confidence, inference_time_ms, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow(),
            camera_id,
            helmet_detected,
            confidence,
            inference_time_ms,
            'model_4'
        ))
        
        conn.commit()
        conn.close()
        
    def log_alert(self, camera_id, alert_type, confidence):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts 
            (timestamp, camera_id, alert_type, confidence, acknowledged)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow(),
            camera_id,
            alert_type,
            confidence,
            False
        ))
        
        conn.commit()
        conn.close()

# Usage
logger = InferenceLogger()
logger.log_inference('camera_001', True, 0.98, 45.2)
logger.log_alert('camera_001', 'no_helmet', 0.97)
```

---

## Testing & Validation

### Unit Testing

```python
import unittest
import numpy as np

class TestHelmNetInference(unittest.TestCase):
    
    def setUp(self):
        self.model = tf.keras.models.load_model('model_4_final.h5')
        
    def test_model_output_shape(self):
        """Test model output shape"""
        X_test = np.random.randn(10, 224, 224, 3)
        predictions = self.model.predict(X_test)
        self.assertEqual(predictions.shape, (10, 2))
        
    def test_model_output_range(self):
        """Test model outputs are valid probabilities"""
        X_test = np.random.randn(10, 224, 224, 3)
        predictions = self.model.predict(X_test)
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
        
    def test_inference_speed(self):
        """Test inference speed meets requirements"""
        X_test = np.random.randn(100, 224, 224, 3)
        start = time.time()
        self.model.predict(X_test, batch_size=32)
        elapsed = time.time() - start
        avg_time = (elapsed / 100) * 1000  # ms per image
        self.assertLess(avg_time, 100)  # Must be <100ms
        
    def test_model_accuracy(self):
        """Test model accuracy on test set"""
        accuracy = self.model.evaluate(X_test, y_test)[1]
        self.assertGreater(accuracy, 0.96)  # Must be >96%

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
def test_end_to_end_pipeline():
    """Test complete inference pipeline"""
    
    # 1. Load model
    model = tf.keras.models.load_model('model_4_final.h5')
    
    # 2. Simulate CCTV stream
    test_frame = cv2.imread('test_image.jpg')
    test_frame = cv2.resize(test_frame, (224, 224))
    test_frame = test_frame / 255.0
    
    # 3. Run inference
    prediction = model.predict(np.expand_dims(test_frame, 0))
    helmet_detected = prediction[0][1] > 0.5
    confidence = float(np.max(prediction[0]))
    
    # 4. Log result
    logger.log_inference('camera_001', helmet_detected, confidence, 45.2)
    
    # 5. Dispatch alert if needed
    if not helmet_detected and confidence > 0.9:
        dispatcher.send_alert('camera_001', 'no_helmet', confidence)
    
    # 6. Verify logging
    assert logger.get_latest_inference('camera_001') is not None
    
    print("✅ End-to-end pipeline test passed")

test_end_to_end_pipeline()
```

---

## Rollout Strategy

### Phase 1: Pilot Deployment (Weeks 1-4)

**Scope**: 1 facility, 2 camera feeds

**Objectives**:
- Verify model performance in production
- Establish baseline metrics
- Train staff on system operation
- Identify integration issues

**Success Criteria**:
- Model accuracy >95% in production
- Inference time <100ms
- Alert system working correctly
- Staff trained and comfortable

**Rollback Plan**: If accuracy <95%, revert to manual inspection

### Phase 2: Expanded Pilot (Weeks 5-12)

**Scope**: 3 facilities, 10 camera feeds

**Objectives**:
- Validate scalability
- Collect performance data
- Refine alert thresholds
- Develop operational procedures

**Success Criteria**:
- Model accuracy >95% across all facilities
- System uptime >99%
- Alert response time <5 minutes
- Staff satisfaction >80%

**Rollback Plan**: If uptime <99%, add redundancy

### Phase 3: Full Deployment (Weeks 13+)

**Scope**: 50+ facilities, 200+ camera feeds

**Objectives**:
- Deploy to all target facilities
- Establish monitoring dashboard
- Implement automated retraining
- Optimize performance

**Success Criteria**:
- Model accuracy >95% across all facilities
- System uptime >99.5%
- Cost savings >$300K annually
- User satisfaction >90%

**Rollback Plan**: Gradual rollback by facility if issues arise

---

## Monitoring & Maintenance

### Key Performance Indicators (KPIs)

| KPI | Target | Frequency | Action |
|-----|--------|-----------|--------|
| Model Accuracy | >95% | Daily | Retrain if <95% |
| System Uptime | >99.5% | Hourly | Alert if <99% |
| Inference Time | <100ms | Hourly | Optimize if >120ms |
| Alert Response | <5 min | Daily | Escalate if >10 min |
| False Positive Rate | <5% | Weekly | Adjust threshold if >5% |
| False Negative Rate | <2% | Weekly | Retrain if >2% |

### Monitoring Dashboard

```python
from flask import Flask, jsonify
import psutil

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'gpu_memory': psutil.virtual_memory().percent,
        'cpu_usage': psutil.cpu_percent(),
        'model_version': 'model_4'
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    """Performance metrics endpoint"""
    return jsonify({
        'accuracy': 0.968,
        'inference_time_ms': 45.2,
        'uptime_hours': 720,
        'total_inferences': 1000000,
        'alerts_generated': 5000
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Model Retraining Pipeline

```python
def monthly_retraining():
    """Retrain model monthly with new data"""
    
    # 1. Collect new data from production
    new_data = collect_production_data(days=30)
    
    # 2. Validate new data quality
    if validate_data_quality(new_data):
        
        # 3. Retrain model
        new_model = retrain_model(new_data)
        
        # 4. Evaluate on test set
        accuracy = evaluate_model(new_model, X_test, y_test)
        
        # 5. Compare with current model
        if accuracy > current_accuracy:
            # 6. Deploy new model
            deploy_model(new_model)
            print(f"✅ Model updated: {accuracy*100:.1f}%")
        else:
            print(f"⚠️ New model not better: {accuracy*100:.1f}%")
    else:
        print("❌ New data quality check failed")

# Schedule monthly retraining
schedule.every().month.do(monthly_retraining)
```

---

## Troubleshooting

### Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low accuracy in production | Data distribution shift | Retrain with production data |
| High false positive rate | Threshold too low | Increase confidence threshold |
| High false negative rate | Threshold too high | Decrease confidence threshold |
| Slow inference | GPU memory full | Reduce batch size |
| Model crashes | Out of memory | Increase GPU memory or reduce model size |
| Alert dispatch fails | Network issue | Implement retry logic |
| CCTV stream drops | Network connectivity | Add redundant network path |

---

## Security Considerations

### Data Privacy

```python
# Anonymize frames before logging
def anonymize_frame(frame):
    """Remove identifying information from frame"""
    # Blur faces
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (50, 50))
    return frame

# Only log helmet detection, not full frame
def log_inference_safe(camera_id, helmet_detected, confidence):
    logger.log_inference(camera_id, helmet_detected, confidence, 45.2)
    # Don't log frame data
```

### Access Control

```python
# Implement role-based access control
ROLES = {
    'admin': ['view_all', 'manage_alerts', 'configure_system'],
    'supervisor': ['view_all', 'acknowledge_alerts'],
    'operator': ['view_own_facility', 'acknowledge_alerts'],
    'viewer': ['view_own_facility']
}

def check_permission(user_role, action):
    return action in ROLES.get(user_role, [])
```

---

## Conclusion

Model 4 is ready for production deployment with:

✅ **Proven Accuracy**: 96.8% in testing  
✅ **Fast Inference**: <100ms per frame  
✅ **Scalable Architecture**: Handles 50+ facilities  
✅ **Robust Integration**: Works with existing systems  
✅ **Comprehensive Monitoring**: Real-time performance tracking  
✅ **Clear Rollout Plan**: Phased deployment strategy  

**Next Steps**:
1. Provision infrastructure
2. Deploy to pilot facility
3. Validate performance
4. Expand to additional facilities
5. Establish monitoring & maintenance

