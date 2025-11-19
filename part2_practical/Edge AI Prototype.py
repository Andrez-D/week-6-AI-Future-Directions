"""
Edge AI Image Classification - Recyclable Items Classifier
Assignment: AI Future Directions - Task 1
Author: [Your Name]
Date: November 2025

This script demonstrates Edge AI by:
1. Training a lightweight image classification model
2. Converting it to TensorFlow Lite
3. Testing on sample data with metrics
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================
# STEP 1: CREATE SAMPLE DATASET (OR LOAD REAL DATA)
# ============================================
# For demonstration, we'll create synthetic data
# Replace this with real dataset from Kaggle (e.g., TrashNet dataset)

def create_sample_dataset(num_samples=1000, img_size=96):
    """
    Creates synthetic dataset for recyclable classification
    Categories: plastic, paper, glass, metal, organic
    
    For production: Use tf.keras.preprocessing.image_dataset_from_directory()
    with a real dataset like TrashNet or Waste Classification
    """
    num_classes = 5
    classes = ['plastic', 'paper', 'glass', 'metal', 'organic']
    
    # Generate synthetic images (replace with real data loading)
    X = np.random.rand(num_samples, img_size, img_size, 3).astype('float32')
    y = np.random.randint(0, num_classes, num_samples)
    
    # Split into train/val/test
    split1, split2 = int(0.7 * num_samples), int(0.85 * num_samples)
    
    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes

# Load or create dataset
print("\n📦 Loading dataset...")
(X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = create_sample_dataset()
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ============================================
# STEP 2: BUILD LIGHTWEIGHT MODEL
# ============================================
# Using MobileNetV2 architecture - optimized for edge devices

def build_edge_model(input_shape=(96, 96, 3), num_classes=5):
    """
    Creates a lightweight CNN optimized for edge deployment
    Uses MobileNetV2 backbone with custom classification head
    """
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=0.5  # Width multiplier for lighter model
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Custom head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model

print("\n🏗️  Building Edge AI model...")
model = build_edge_model(num_classes=len(class_names))

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print(f"\n📊 Total parameters: {model.count_params():,}")

# ============================================
# STEP 3: TRAIN MODEL
# ============================================

print("\n🎯 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# ============================================
# STEP 4: EVALUATE MODEL
# ============================================

print("\n📈 Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved confusion_matrix.png")

# ============================================
# STEP 5: CONVERT TO TENSORFLOW LITE
# ============================================

print("\n🔄 Converting to TensorFlow Lite...")

# Standard conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
tflite_model_path = 'recyclable_classifier.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

tflite_size_kb = len(tflite_model) / 1024
print(f"✅ TFLite model saved: {tflite_model_path}")
print(f"📦 Model size: {tflite_size_kb:.2f} KB")

# Quantized conversion for even smaller size
print("\n🗜️  Creating quantized version...")
converter_quantized = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quantized.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter_quantized.convert()

tflite_quantized_path = 'recyclable_classifier_quantized.tflite'
with open(tflite_quantized_path, 'wb') as f:
    f.write(tflite_quantized_model)

quantized_size_kb = len(tflite_quantized_model) / 1024
print(f"✅ Quantized model saved: {tflite_quantized_path}")
print(f"📦 Quantized size: {quantized_size_kb:.2f} KB")
print(f"🎯 Size reduction: {((tflite_size_kb - quantized_size_kb) / tflite_size_kb * 100):.1f}%")

# ============================================
# STEP 6: TEST TFLITE MODEL
# ============================================

def test_tflite_model(tflite_path, X_test, y_test):
    """Test TFLite model and measure inference time"""
    import time
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on subset
    num_test = min(100, len(X_test))
    correct = 0
    inference_times = []
    
    for i in range(num_test):
        # Prepare input
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        start = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start) * 1000  # ms
        inference_times.append(inference_time)
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_class = np.argmax(output_data)
        
        if pred_class == y_test[i]:
            correct += 1
    
    accuracy = correct / num_test
    avg_inference = np.mean(inference_times)
    
    return accuracy, avg_inference

print("\n⚡ Testing TFLite model performance...")
lite_accuracy, lite_inference = test_tflite_model(tflite_model_path, X_test, y_test)
print(f"TFLite Accuracy: {lite_accuracy:.4f}")
print(f"Average inference time: {lite_inference:.2f} ms")

print("\n⚡ Testing Quantized TFLite model...")
quant_accuracy, quant_inference = test_tflite_model(tflite_quantized_path, X_test, y_test)
print(f"Quantized Accuracy: {quant_accuracy:.4f}")
print(f"Average inference time: {quant_inference:.2f} ms")

# ============================================
# STEP 7: DEPLOYMENT METRICS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 EDGE AI DEPLOYMENT SUMMARY")
print("="*60)
print(f"Original Model Accuracy: {test_accuracy:.4f}")
print(f"TFLite Model Accuracy: {lite_accuracy:.4f}")
print(f"Quantized Model Accuracy: {quant_accuracy:.4f}")
print(f"\nModel Size:")
print(f"  - Standard TFLite: {tflite_size_kb:.2f} KB")
print(f"  - Quantized TFLite: {quantized_size_kb:.2f} KB")
print(f"\nInference Speed:")
print(f"  - Standard TFLite: {lite_inference:.2f} ms")
print(f"  - Quantized TFLite: {quant_inference:.2f} ms")
print("="*60)

print("\n✅ Edge AI model ready for deployment!")
print("\n📝 BENEFITS OF EDGE AI:")
print("1. ⚡ Low Latency: ~{:.2f}ms inference (vs 100-500ms cloud)".format(lite_inference))
print("2. 🔒 Privacy: Data processed locally, no cloud upload")
print("3. 🌐 Offline: Works without internet connection")
print("4. 💰 Cost: No cloud API fees")
print("5. 📦 Lightweight: {:.2f}KB model fits on IoT devices".format(quantized_size_kb))

# ============================================
# DEPLOYMENT INSTRUCTIONS
# ============================================

deployment_code = """
# ============================================
# RASPBERRY PI DEPLOYMENT EXAMPLE
# ============================================

# Install dependencies:
# pip install tflite-runtime pillow numpy

import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

# Load model
interpreter = tflite.Interpreter(model_path='recyclable_classifier_quantized.tflite')
interpreter.allocate_tensors()

# Get input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Process image
def classify_image(image_path):
    # Load and preprocess
    img = Image.open(image_path).resize((96, 96))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Result
    classes = ['plastic', 'paper', 'glass', 'metal', 'organic']
    pred_class = classes[np.argmax(output)]
    confidence = np.max(output)
    
    return pred_class, confidence

# Use in real-time
result, conf = classify_image('recyclable_item.jpg')
print(f"Detected: {result} ({conf:.2%} confidence)")
"""

with open('deployment_example.py', 'w') as f:
    f.write(deployment_code)

print("\n💾 Saved deployment_example.py for Raspberry Pi usage")
print("\n✨ Task 1 Complete! Ready for GitHub submission.")
