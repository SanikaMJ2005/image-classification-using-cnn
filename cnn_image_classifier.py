"""
Simple CNN Model for Image Classification
Focus: Data Loading, Model Training, and Evaluation
Dataset: MNIST (handwritten digits 0-9)
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("=" * 70)
print("1. LOADING AND PREPROCESSING DATA")
print("=" * 70)

# Load MNIST dataset (70,000 images of 28x28 handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Original Training Data Shape: {x_train.shape}")
print(f"Original Test Data Shape: {x_test.shape}")

# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to add channel dimension (for grayscale: 1 channel)
# From (num_samples, 28, 28) to (num_samples, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels (0 -> [1,0,0,...,0], 1 -> [0,1,0,...,0], etc.)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Processed Training Data Shape: {x_train.shape}")
print(f"Processed Test Data Shape: {x_test.shape}")
print(f"Processed Labels Shape: {y_train.shape}")

# Visualize some sample images
print("\nSample images from dataset:")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {np.argmax(y_train[i])}")
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png', dpi=100, bbox_inches='tight')
print("✓ Saved 'sample_images.png'")
plt.close()

# ============================================================================
# 2. MODEL BUILDING
# ============================================================================
print("\n" + "=" * 70)
print("2. BUILDING CNN MODEL")
print("=" * 70)

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(10, activation='softmax')  # Output layer (10 classes)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============================================================================
# 3. MODEL TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("3. TRAINING THE MODEL")
print("=" * 70)

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 70)
print("4. EVALUATING THE MODEL")
print("=" * 70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification Report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

# Plot 1: Training and Validation Accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Over Epochs')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Model Loss Over Epochs')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
print("✓ Saved 'training_history.png'")
plt.close()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
print("✓ Saved 'confusion_matrix.png'")
plt.close()

# Plot 3: Correct and Incorrect Predictions
correct_indices = np.where(y_pred == y_true)[0]
incorrect_indices = np.where(y_pred != y_true)[0]

fig, axes = plt.subplots(2, 5, figsize=(14, 6))

# Show correct predictions
for i, ax in enumerate(axes[0]):
    idx = correct_indices[i]
    ax.imshow(x_test[idx].squeeze(), cmap='gray')
    ax.set_title(f"✓ True: {y_true[idx]}", color='green')
    ax.axis('off')

# Show incorrect predictions
for i, ax in enumerate(axes[1]):
    idx = incorrect_indices[i]
    ax.imshow(x_test[idx].squeeze(), cmap='gray')
    ax.set_title(f"✗ True: {y_true[idx]}, Pred: {y_pred[idx]}", color='red')
    ax.axis('off')

plt.suptitle('Sample Predictions: Correct (Top) vs Incorrect (Bottom)')
plt.tight_layout()
plt.savefig('predictions_sample.png', dpi=100, bbox_inches='tight')
print("✓ Saved 'predictions_sample.png'")
plt.close()

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Model trained on {len(x_train)} images")
print(f"Model evaluated on {len(x_test)} images")
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Total Correct Predictions: {len(correct_indices)}")
print(f"Total Incorrect Predictions: {len(incorrect_indices)}")
print("\nGenerated Visualizations:")
print("  1. sample_images.png - Sample images from the dataset")
print("  2. training_history.png - Training and validation curves")
print("  3. confusion_matrix.png - Confusion matrix heatmap")
print("  4. predictions_sample.png - Sample correct and incorrect predictions")
print("=" * 70)
