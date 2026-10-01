# Databricks notebook source
# DBTITLE 1,Install and Import Packages
# Install TensorFlow if not already available
%pip install tensorflow matplotlib numpy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print("✓ Packages imported successfully")

# COMMAND ----------

# DBTITLE 1,Load and Prepare Data (3 Classes)
# Load CIFAR-10 dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()

# CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
#                   5=dog, 6=frog, 7=horse, 8=ship, 9=truck

# Select 3 classes for our example: airplane (0), automobile (1), bird (2)
selected_classes = [0, 1, 2]
class_names = ['airplane', 'automobile', 'bird']

# Filter training data
train_mask = np.isin(y_train_full, selected_classes).flatten()
X_train = X_train_full[train_mask]
y_train = y_train_full[train_mask]

# Filter test data
test_mask = np.isin(y_test_full, selected_classes).flatten()
X_test = X_test_full[test_mask]
y_test = y_test_full[test_mask]

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
# This is REQUIRED for categorical_crossentropy loss
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

print(f"\n📊 Dataset Statistics:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Image shape: {X_train.shape[1:]}")
print(f"Number of classes: {len(selected_classes)}")
print(f"\nLabel distribution (training):")
for i, class_name in enumerate(class_names):
    count = np.sum(y_train == i)
    print(f"  {class_name}: {count} samples")
print(f"\ny_train shape: {y_train.shape}")
print(f"y_train_cat shape (one-hot): {y_train_cat.shape}")
print(f"\nSample one-hot encoding:")
print(f"  Original label: {y_train[0][0]} ({class_names[y_train[0][0]]})")
print(f"  One-hot vector: {y_train_cat[0]}")

# COMMAND ----------

# DBTITLE 1,Visualize Sample Images
# Display sample images from each class
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
fig.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')

for class_idx in range(3):
    # Get 5 random samples from this class
    class_samples = X_train[y_train.flatten() == class_idx]
    random_indices = np.random.choice(len(class_samples), 5, replace=False)
    
    for i, img_idx in enumerate(random_indices):
        axes[class_idx, i].imshow(class_samples[img_idx])
        axes[class_idx, i].axis('off')
        if i == 0:
            axes[class_idx, i].set_ylabel(class_names[class_idx], 
                                          fontsize=12, 
                                          fontweight='bold',
                                          rotation=0,
                                          ha='right',
                                          va='center')

plt.tight_layout()
plt.show()

print("✓ Sample images displayed")

# COMMAND ----------

# DBTITLE 1,Build CNN Model
# Build CNN: Conv2D → MaxPool → Flatten → Dense → Softmax

model = keras.Sequential([
    # Input layer (32x32x3 images)
    layers.Input(shape=(32, 32, 3)),
    
    # Convolutional layer 1 + MaxPooling
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional layer 2 + MaxPooling
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Convolutional layer 3 + MaxPooling
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten before dense layers
    layers.Flatten(),
    
    # Dense layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    
    # Output layer: 3 neurons with SOFTMAX (NOT sigmoid!)
    # Softmax is for multi-class classification
    layers.Dense(3, activation='softmax')
], name='3_class_cnn')

# Display model architecture
model.summary()

print("\n🏗️ Model Architecture:")
print("  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → MaxPool")
print("  → Flatten → Dense(64) → Dropout → Dense(3, softmax)")
print("\n⚠️ Important: Using SOFTMAX (not sigmoid) for 3-class classification")

# COMMAND ----------

# DBTITLE 1,Compile Model
# Compile model
# Loss: categorical_crossentropy (for multi-class with one-hot labels)
# Optimizer: Adam
# Metrics: accuracy

model.compile(
    loss='categorical_crossentropy',  # For multi-class classification
    optimizer='adam',                  # Adam optimizer
    metrics=['accuracy']               # Track accuracy during training
)

print("✓ Model compiled")
print("\n📋 Compilation settings:")
print(f"  Loss: categorical_crossentropy")
print(f"  Optimizer: Adam")
print(f"  Metrics: accuracy")
print("\n💡 Note: categorical_crossentropy requires one-hot encoded labels")
print("   (If using integer labels, use sparse_categorical_crossentropy instead)")

# COMMAND ----------

# DBTITLE 1,Train Model with Validation
# Train the model with validation split to monitor overfitting

print("🚀 Starting training...\n")

history = model.fit(
    X_train, 
    y_train_cat,
    batch_size=64,              # Train with batches
    epochs=20,                  # Train for 20 epochs
    validation_split=0.2,       # Use 20% of training data for validation
    verbose=1,                  # Show progress bar
    shuffle=True                # Shuffle training data each epoch
)

print("\n✅ Training complete!")
print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

# COMMAND ----------

# DBTITLE 1,Plot Training History (Detect Overfitting)
# Plot training & validation accuracy and loss to detect overfitting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print("\n📊 Overfitting Analysis:")
print(f"  Accuracy gap: {(train_acc - val_acc)*100:.2f}%")
print(f"  Loss gap: {(val_loss - train_loss):.4f}")

if train_acc - val_acc > 0.1:
    print("  ⚠️ Model is OVERFITTING (training accuracy >> validation accuracy)")
    print("  💡 Solutions: Add more dropout, use data augmentation, reduce model complexity")
elif train_acc - val_acc > 0.05:
    print("  ⚡ Slight overfitting detected")
    print("  💡 Consider adding regularization or early stopping")
else:
    print("  ✅ Model is generalizing well (minimal overfitting)")

# COMMAND ----------

# DBTITLE 1,Evaluate on Test Set
# Evaluate the model on the test set

print("🧪 Evaluating on test set...\n")

test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

print(f"\n📈 Test Results:")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")

# Get predictions for confusion analysis
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# Per-class accuracy
print(f"\n📊 Per-Class Accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = (y_true == i)
    class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
    print(f"  {class_name}: {class_acc*100:.2f}%")

# COMMAND ----------

# DBTITLE 1,Make Predictions on Sample Images
# Make predictions on random test images

# Select 9 random test images
random_indices = np.random.choice(len(X_test), 9, replace=False)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Model Predictions on Test Images', fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    img_idx = random_indices[idx]
    
    # Get image and prediction
    img = X_test[img_idx]
    true_label = y_test[img_idx][0]
    
    # Make prediction
    pred_probs = model.predict(img.reshape(1, 32, 32, 3), verbose=0)[0]
    pred_label = np.argmax(pred_probs)
    
    # Display image
    ax.imshow(img)
    
    # Set title with prediction and confidence
    title = f"True: {class_names[true_label]}\n"
    title += f"Pred: {class_names[pred_label]} ({pred_probs[pred_label]*100:.1f}%)"
    
    # Color: green if correct, red if wrong
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\n✅ Predictions displayed")
print("   Green = Correct, Red = Incorrect")

# COMMAND ----------

# DBTITLE 1,Summary and Key Takeaways
print("="*80)
print("🎓 3-CLASS CNN CLASSIFICATION - KEY TAKEAWAYS")
print("="*80)

print("\n1️⃣ ARCHITECTURE:")
print("   Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling")
print("   → Flatten → Dense → Dropout → Dense(3, softmax)")

print("\n2️⃣ ACTIVATION FUNCTIONS:")
print("   • Hidden layers: ReLU")
print("   • Output layer: SOFTMAX (NOT sigmoid!)")
print("   • Softmax is for multi-class (outputs probability distribution)")
print("   • Sigmoid is for binary classification only")

print("\n3️⃣ LOSS FUNCTION:")
print("   • categorical_crossentropy (requires one-hot encoded labels)")
print("   • Alternative: sparse_categorical_crossentropy (for integer labels)")

print("\n4️⃣ OPTIMIZER:")
print("   • Adam (adaptive learning rate, good default choice)")

print("\n5️⃣ TRAINING:")
print("   • Batch size: 64 (process 64 images at a time)")
print("   • Validation split: 20% (monitor overfitting)")
print("   • Shuffle: True (randomize training data each epoch)")

print("\n6️⃣ OVERFITTING DETECTION:")
print("   • Monitor validation accuracy vs training accuracy")
print("   • If val_acc << train_acc → overfitting")
print("   • Solutions: dropout, data augmentation, regularization, early stopping")

print("\n7️⃣ MODEL EVALUATION:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Test Loss: {test_loss:.4f}")

print("\n💡 TIPS FOR BETTER PERFORMANCE:")
print("   • Add data augmentation (rotation, flipping, zoom)")
print("   • Use batch normalization after Conv2D layers")
print("   • Implement early stopping (stop when val_loss stops improving)")
print("   • Try different architectures (more/fewer layers, different filter sizes)")
print("   • Experiment with learning rate schedules")

print("\n" + "="*80)
print("✅ Tutorial Complete!")
print("="*80)