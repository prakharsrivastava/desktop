# Databricks notebook source
# DBTITLE 1,Fix matplotlib compatibility
# MAGIC %pip install numpy==1.24.3

# COMMAND ----------

# DBTITLE 1,Interactive Parameters
# MAGIC %md
# MAGIC # Interactive CNN Training & Visualization
# MAGIC
# MAGIC This notebook allows you to train a CNN on CIFAR-10 with **interactive parameters** and **visualize what happens inside the network**!
# MAGIC
# MAGIC ## Training Parameters:
# MAGIC * 🎯 **CIFAR-10 Classes to Train On** - Select which classes to include (1-10 classes)
# MAGIC * 📦 **Batch Size** - Training batch size (32, 64, 128, 256)
# MAGIC * ⏱️ **Number of Epochs** - How long to train (10, 20, 30, 50)
# MAGIC * 💧 **Dropout Rate** - Regularization strength (0.3-0.6)
# MAGIC * 📊 **Validation Split** - Percentage for validation (10-30%)
# MAGIC
# MAGIC ## Layer Visualization Parameters:
# MAGIC * 🔬 **Layer to Visualize** - Choose which CNN layer to inspect
# MAGIC * 🎨 **Filter/Channel Number** - Select which filter to view (0-31)
# MAGIC * 🖼️ **Test Image Index** - Pick which test image to analyze (0-49)
# MAGIC
# MAGIC **To experiment:** Change the parameters above and re-run the cells! The model and visualizations adapt automatically.

# COMMAND ----------

# DBTITLE 1,Import Libraries
# Install TensorFlow compatible with pandas 1.5.3 (requires numpy<2)
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow", "numpy<2"])

# Restart Python to ensure clean imports
dbutils.library.restartPython()

# Note: After restart, run the next cell to import packages

# COMMAND ----------

# DBTITLE 1,Import Packages
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

# DBTITLE 1,Load CIFAR-10 Dataset
# Load CIFAR-10 dataset
print("📥 Loading CIFAR-10 dataset...")
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
print(f"✓ Dataset loaded: {len(X_train_full)} training samples, {len(X_test_full)} test samples")

# COMMAND ----------

# DBTITLE 1,Configure Classes (Interactive)
# CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 
#                   5=dog, 6=frog, 7=horse, 8=ship, 9=truck

# Parse selected classes from widget
class_mapping = {
    '0 - airplane': (0, 'airplane'),
    '1 - automobile': (1, 'automobile'),
    '2 - bird': (2, 'bird'),
    '3 - cat': (3, 'cat'),
    '4 - deer': (4, 'deer'),
    '5 - dog': (5, 'dog'),
    '6 - frog': (6, 'frog'),
    '7 - horse': (7, 'horse'),
    '8 - ship': (8, 'ship'),
    '9 - truck': (9, 'truck')
}

# Get selected classes from parameter
selected_str = dbutils.widgets.get('selected_classes')
selected_items = [s.strip() for s in selected_str.split(',')]

selected_classes = [class_mapping[item][0] for item in selected_items]
class_names = [class_mapping[item][1] for item in selected_items]

print(f"✓ Training on {len(selected_classes)} classes: {', '.join(class_names)}")
print(f"  Class indices: {selected_classes}")

# COMMAND ----------

# DBTITLE 1,Filter and Normalize Data
# Filter data to selected classes
print(f"
🔍 Filtering data for classes: {', '.join(class_names)}")

train_mask = np.isin(y_train_full, selected_classes).flatten()
X_train = X_train_full[train_mask]
y_train = y_train_full[train_mask]

test_mask = np.isin(y_test_full, selected_classes).flatten()
X_test = X_test_full[test_mask]
y_test = y_test_full[test_mask]

print(f"✓ Filtered to {len(X_train)} training and {len(X_test)} test samples")

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("✓ Normalized pixel values to [0, 1]")

# COMMAND ----------

# DBTITLE 1,One-Hot Encode Labels
# Convert labels to categorical (one-hot encoding)
# This is REQUIRED for categorical_crossentropy loss
num_classes = len(selected_classes)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

print(f"✓ Labels converted to one-hot encoding with {num_classes} classes")

# COMMAND ----------

# DBTITLE 1,Load and Prepare Data (3 Classes)
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
num_classes = len(selected_classes)
fig, axes = plt.subplots(num_classes, 5, figsize=(12, num_classes * 2.5))
fig.suptitle(f'Sample Images from {num_classes} Classes', fontsize=16, fontweight='bold')

# Handle case where num_classes = 1 (axes won't be 2D)
if num_classes == 1:
    axes = axes.reshape(1, -1)

for class_idx in range(num_classes):
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
# Get hyperparameters from widgets
dropout = float(dbutils.widgets.get('dropout_rate'))
num_classes = len(selected_classes)

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
    layers.Dropout(dropout),  # Dropout to reduce overfitting
    
    # Output layer: softmax for multi-class classification
    layers.Dense(num_classes, activation='softmax')
], name=f'{num_classes}_class_cnn')

# Display model architecture
model.summary()

print("\n🏗️ Model Architecture:")
print("  Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(64) → MaxPool")
print(f"  → Flatten → Dense(64) → Dropout({dropout}) → Dense({num_classes}, softmax)")
print(f"\n✨ Hyperparameters: {num_classes} classes, {dropout} dropout rate")

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
# Get training hyperparameters from widgets
batch_size = int(dbutils.widgets.get('batch_size'))
num_epochs = int(dbutils.widgets.get('num_epochs'))
val_split = float(dbutils.widgets.get('validation_split'))

print("🚀 Starting training...")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {num_epochs}")
print(f"  Validation split: {val_split*100:.0f}%\n")

history = model.fit(
    X_train, 
    y_train_cat,
    batch_size=batch_size,
    epochs=num_epochs,
    validation_split=val_split,
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
print(f"🎓 INTERACTIVE CNN CLASSIFICATION - KEY TAKEAWAYS ({num_classes} Classes)")
print("="*80)

print("
🎮 INTERACTIVE PARAMETERS:")
print(f"   • Classes: {', '.join(class_names)}")
print(f"   • Batch size: {batch_size}")
print(f"   • Epochs: {num_epochs}")
print(f"   • Dropout: {dropout}")
print(f"   • Validation split: {val_split*100:.0f}%")

print("
1️⃣ ARCHITECTURE:")
print("   Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling")
print(f"   → Flatten → Dense → Dropout({dropout}) → Dense({num_classes}, softmax)")

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

print("\n5️⃣ TRAINING (CUSTOMIZABLE):")
print(f"   • Batch size: {batch_size} (adjust via widget)")
print(f"   • Epochs: {num_epochs} (adjust via widget)")
print(f"   • Validation split: {val_split*100:.0f}% (adjust via widget)")
print("   • Shuffle: True (randomize training data each epoch)")

print("\n6️⃣ OVERFITTING DETECTION:")
print("   • Monitor validation accuracy vs training accuracy")
print("   • If val_acc << train_acc → overfitting")
print("   • Solutions: dropout, data augmentation, regularization, early stopping")

print("\n7️⃣ MODEL EVALUATION:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Test Loss: {test_loss:.4f}")

print("\n💡 EXPERIMENT IDEAS:")
print("   • Try different class combinations (use the widget!)")
print("   • Increase dropout if overfitting (0.5 → 0.6)")
print("   • Train longer (20 → 30 or 50 epochs)")
print("   • Adjust batch size (smaller = more updates, larger = faster)")
print("   • Explore layer visualizations below to see what the CNN learns!")
print("   • Compare filters across different layers")
print("   • Try training on all 10 classes!")

print("\n" + "="*80)
print("✅ Tutorial Complete!")
print("="*80)

# COMMAND ----------

# DBTITLE 1,🔬 Layer Visualization Section
# MAGIC %md
# MAGIC # 🔬 Interactive Layer Visualization
# MAGIC
# MAGIC Explore what happens to images as they pass through each layer of the CNN!
# MAGIC
# MAGIC ## Controls:
# MAGIC * 🎯 **Layer to Visualize** - Select which layer to inspect
# MAGIC * 🎨 **Filter/Channel Number** - Choose which filter's output to view
# MAGIC * 🖼️ **Test Image Index** - Pick which test image to analyze
# MAGIC
# MAGIC **How it works:**
# MAGIC * Convolutional layers extract features (edges, textures, patterns)
# MAGIC * Each filter learns to detect different features
# MAGIC * Deeper layers detect more complex patterns
# MAGIC * Pooling layers reduce spatial dimensions while preserving features
# MAGIC
# MAGIC **Try this:** Start with "Conv2D Layer 1" and cycle through different filters to see what each one detects!

# COMMAND ----------

# DBTITLE 1,Create Feature Extraction Model
# Create a model that outputs intermediate layer activations
# This lets us see what the CNN "sees" at each layer

from tensorflow.keras.models import Model

# Get all layer outputs (skip the input layer)
layer_outputs = [layer.output for layer in model.layers[1:]]  # Exclude input layer

# Create a model that returns activations from all layers
feature_extraction_model = Model(inputs=model.input, outputs=layer_outputs)

print("✓ Feature extraction model created")
print(f"
Extracting outputs from {len(layer_outputs)} layers:")
for i, layer in enumerate(model.layers[1:]):
    print(f"  {i}: {layer.name} - Output shape: {layer.output.shape}")

# COMMAND ----------

# DBTITLE 1,Interactive Layer Visualization
# Interactive visualization of layer outputs
import numpy as np
from PIL import Image
import io

# Get parameters from widgets
selected_layer = dbutils.widgets.get('viz_layer')
filter_idx = int(dbutils.widgets.get('viz_filter'))
image_idx = int(dbutils.widgets.get('viz_image_idx'))

# Get a test image
if image_idx >= len(X_test):
    image_idx = 0
test_image = X_test[image_idx]
true_label = y_test[image_idx][0]

# Map layer names to indices (1-indexed because we skip input)
layer_map = {
    'Input Image': -1,
    'Conv2D Layer 1': 0,  # conv2d
    'MaxPool 1': 1,       # max_pooling2d
    'Conv2D Layer 2': 2,  # conv2d_1
    'MaxPool 2': 3,       # max_pooling2d_1
    'Conv2D Layer 3': 4,  # conv2d_2
    'MaxPool 3': 5        # max_pooling2d_2
}

if selected_layer == 'Input Image':
    # Show the original image using PIL and display
    print(f"🖼️ Original 32x32x3 RGB Image")
    print(f"True Label: {class_names[true_label]}")
    print(f"Image Index: {image_idx}")
    print("\n" + "="*60)
    
    # Convert to PIL Image and resize for better visibility
    img_uint8 = (test_image * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    pil_img_large = pil_img.resize((256, 256), Image.NEAREST)  # Make it bigger
    
    display(pil_img_large)
    
    print(f"\n📊 Image Statistics:")
    print(f"  Shape: {test_image.shape}")
    print(f"  Value range: [{test_image.min():.3f}, {test_image.max():.3f}]")
    print(f"  Mean: R={test_image[:,:,0].mean():.3f}, G={test_image[:,:,1].mean():.3f}, B={test_image[:,:,2].mean():.3f}")
else:
    # Get activations for all layers
    activations = feature_extraction_model.predict(test_image.reshape(1, 32, 32, 3), verbose=0)
    
    # Get the selected layer's activation
    layer_idx = layer_map[selected_layer]
    layer_activation = activations[layer_idx][0]  # Remove batch dimension
    
    layer_name = model.layers[layer_idx + 1].name
    
    # Determine if this is a conv or pooling layer
    is_conv = 'conv' in layer_name.lower()
    
    if is_conv:
        # For conv layers, show individual filters
        num_filters = layer_activation.shape[-1]
        
        # Ensure filter_idx is valid
        if filter_idx >= num_filters:
            filter_idx = 0
        
        print(f"🔍 {selected_layer} - Filter {filter_idx}")
        print(f"Layer: {layer_name}")
        print(f"Output shape: {layer_activation.shape}")
        print(f"Number of filters: {num_filters}")
        print("\n" + "="*60)
        
        # 1. Original Image
        print("\n📸 1. ORIGINAL IMAGE")
        img_uint8 = (test_image * 255).astype(np.uint8)
        pil_orig = Image.fromarray(img_uint8)
        pil_orig_large = pil_orig.resize((200, 200), Image.NEAREST)
        display(pil_orig_large)
        print(f"True Label: {class_names[true_label]}")
        
        # 2. Selected filter output
        print(f"\n🎨 2. FILTER {filter_idx} OUTPUT (what this filter detects)")
        feature_map = layer_activation[:, :, filter_idx]
        
        # Normalize to 0-255 for visualization
        feat_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        feat_uint8 = (feat_normalized * 255).astype(np.uint8)
        
        # Apply colormap manually (viridis-like: blue to green to yellow)
        # Create RGB image
        feat_colored = np.zeros((feat_uint8.shape[0], feat_uint8.shape[1], 3), dtype=np.uint8)
        feat_colored[:, :, 0] = np.clip(feat_uint8 * 1.5 - 100, 0, 255)  # Red
        feat_colored[:, :, 1] = feat_uint8  # Green
        feat_colored[:, :, 2] = np.clip(255 - feat_uint8, 0, 255)  # Blue
        
        pil_feat = Image.fromarray(feat_colored)
        scale_factor = 256 // feature_map.shape[0]
        pil_feat_large = pil_feat.resize((feature_map.shape[0] * scale_factor, 
                                          feature_map.shape[1] * scale_factor), 
                                         Image.NEAREST)
        display(pil_feat_large)
        
        print(f"  Shape: {feature_map.shape}")
        print(f"  Value range: [{feature_map.min():.2f}, {feature_map.max():.2f}]")
        print(f"  Blue = low activation, Yellow/Red = high activation")
        
        # 3. Grid of all filters
        print(f"\n🎯 3. FIRST 16 FILTERS OVERVIEW (Filter {filter_idx} is active)")
        n_display = min(16, num_filters)
        grid_size = int(np.ceil(np.sqrt(n_display)))
        
        # Create montage
        montage = np.zeros((grid_size * layer_activation.shape[0], 
                           grid_size * layer_activation.shape[1]), dtype=np.uint8)
        
        for i in range(n_display):
            row = i // grid_size
            col = i % grid_size
            filter_img = layer_activation[:, :, i]
            # Normalize each filter
            filter_norm = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
            filter_uint8 = (filter_norm * 255).astype(np.uint8)
            montage[row * layer_activation.shape[0]:(row + 1) * layer_activation.shape[0],
                   col * layer_activation.shape[1]:(col + 1) * layer_activation.shape[1]] = filter_uint8
        
        # Apply colormap to montage
        montage_colored = np.zeros((montage.shape[0], montage.shape[1], 3), dtype=np.uint8)
        montage_colored[:, :, 0] = np.clip(montage * 1.5 - 100, 0, 255)
        montage_colored[:, :, 1] = montage
        montage_colored[:, :, 2] = np.clip(255 - montage, 0, 255)
        
        pil_montage = Image.fromarray(montage_colored)
        pil_montage_large = pil_montage.resize((400, 400), Image.NEAREST)
        display(pil_montage_large)
        
        print(f"  Showing {n_display}/{num_filters} filters")
        print(f"  Each small square is one filter's output")
        
        print(f"\n💡 What you're seeing:")
        print(f"  • Bright/Yellow areas: Features STRONGLY detected by this filter")
        print(f"  • Dark/Blue areas: Features NOT detected or suppressed")
        print(f"  • Each filter learns to detect different patterns!")
        print(f"  • Try filters 0-{num_filters-1} to see what each one finds!")
    else:
        # For pooling layers
        num_channels = layer_activation.shape[-1]
        if filter_idx >= num_channels:
            filter_idx = 0
        
        print(f"📐 {selected_layer}")
        print(f"Layer: {layer_name}")
        print(f"Output shape: {layer_activation.shape}")
        print(f"Spatial reduction: {test_image.shape[:2]} → {layer_activation.shape[:2]}")
        print("\n" + "="*60)
        
        # 1. Original Image
        print("\n📸 1. ORIGINAL IMAGE")
        img_uint8 = (test_image * 255).astype(np.uint8)
        pil_orig = Image.fromarray(img_uint8)
        pil_orig_large = pil_orig.resize((200, 200), Image.NEAREST)
        display(pil_orig_large)
        print(f"True Label: {class_names[true_label]}")
        
        # 2. Average pooled output
        print(f"\n🔽 2. AVERAGE ACROSS ALL {num_channels} CHANNELS")
        avg_activation = np.mean(layer_activation, axis=-1)
        
        # Normalize and colorize
        avg_norm = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-8)
        avg_uint8 = (avg_norm * 255).astype(np.uint8)
        
        avg_colored = np.zeros((avg_uint8.shape[0], avg_uint8.shape[1], 3), dtype=np.uint8)
        avg_colored[:, :, 0] = np.clip(avg_uint8 * 1.5 - 100, 0, 255)
        avg_colored[:, :, 1] = avg_uint8
        avg_colored[:, :, 2] = np.clip(255 - avg_uint8, 0, 255)
        
        pil_avg = Image.fromarray(avg_colored)
        scale_factor = 256 // avg_activation.shape[0]
        pil_avg_large = pil_avg.resize((avg_activation.shape[0] * scale_factor, 
                                        avg_activation.shape[1] * scale_factor), 
                                       Image.NEAREST)
        display(pil_avg_large)
        print(f"  Shape: {avg_activation.shape}")
        print(f"  Notice: Image is smaller (downsampled by pooling)")
        
        # 3. Single channel
        print(f"\n🎨 3. CHANNEL {filter_idx} (individual feature map)")
        channel_activation = layer_activation[:, :, filter_idx]
        
        chan_norm = (channel_activation - channel_activation.min()) / (channel_activation.max() - channel_activation.min() + 1e-8)
        chan_uint8 = (chan_norm * 255).astype(np.uint8)
        
        chan_colored = np.zeros((chan_uint8.shape[0], chan_uint8.shape[1], 3), dtype=np.uint8)
        chan_colored[:, :, 0] = np.clip(chan_uint8 * 1.5 - 100, 0, 255)
        chan_colored[:, :, 1] = chan_uint8
        chan_colored[:, :, 2] = np.clip(255 - chan_uint8, 0, 255)
        
        pil_chan = Image.fromarray(chan_colored)
        pil_chan_large = pil_chan.resize((channel_activation.shape[0] * scale_factor, 
                                          channel_activation.shape[1] * scale_factor), 
                                         Image.NEAREST)
        display(pil_chan_large)
        print(f"  Shape: {channel_activation.shape}")
        
        print(f"\n💡 MaxPooling Effect:")
        print(f"  • Reduces spatial size: keeps strongest features")
        print(f"  • Makes network robust to small image shifts")
        print(f"  • No parameters to learn (just takes max)")
        print(f"  • Blue = weak features, Yellow/Red = strong features")

# COMMAND ----------

# DBTITLE 1,Complete Layer-by-Layer Transformation
# Visualize how the image transforms through ALL layers
# This shows the complete journey from input to features

import numpy as np
from PIL import Image

# Get image index from widget
image_idx = int(dbutils.widgets.get('viz_image_idx'))
if image_idx >= len(X_test):
    image_idx = 0

test_image = X_test[image_idx]
true_label = y_test[image_idx][0]
pred_probs = model.predict(test_image.reshape(1, 32, 32, 3), verbose=0)[0]
pred_label = np.argmax(pred_probs)

# Get all layer activations
activations = feature_extraction_model.predict(test_image.reshape(1, 32, 32, 3), verbose=0)

# Select only Conv2D and MaxPooling layers (first 6 layers)
conv_pool_layers = [
    ('Input', test_image, (32, 32)),
    ('Conv2D-1', activations[0][0], activations[0][0].shape[:2]),
    ('MaxPool-1', activations[1][0], activations[1][0].shape[:2]),
    ('Conv2D-2', activations[2][0], activations[2][0].shape[:2]),
    ('MaxPool-2', activations[3][0], activations[3][0].shape[:2]),
    ('Conv2D-3', activations[4][0], activations[4][0].shape[:2]),
    ('MaxPool-3', activations[5][0], activations[5][0].shape[:2])
]

# Display complete transformation pipeline
print("="*70)
print(f"🌊 COMPLETE TRANSFORMATION PIPELINE")
print(f"True: {class_names[true_label]} | Predicted: {class_names[pred_label]} ({pred_probs[pred_label]*100:.1f}%)")
print("="*70)

# Create a combined visualization
images_to_display = []

for idx, (layer_name, layer_output, spatial_size) in enumerate(conv_pool_layers):
    if idx == 0:
        # Original RGB image
        img_uint8 = (test_image * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        # Resize to standard size for comparison
        pil_img_resized = pil_img.resize((128, 128), Image.NEAREST)
        images_to_display.append((layer_name, pil_img_resized, f'{spatial_size[0]}x{spatial_size[1]}x3 (RGB)'))
    else:
        # Average across channels and colorize
        avg_output = np.mean(layer_output, axis=-1)
        num_channels = layer_output.shape[-1]
        
        # Normalize
        avg_norm = (avg_output - avg_output.min()) / (avg_output.max() - avg_output.min() + 1e-8)
        avg_uint8 = (avg_norm * 255).astype(np.uint8)
        
        # Apply colormap
        avg_colored = np.zeros((avg_uint8.shape[0], avg_uint8.shape[1], 3), dtype=np.uint8)
        avg_colored[:, :, 0] = np.clip(avg_uint8 * 1.5 - 100, 0, 255)
        avg_colored[:, :, 1] = avg_uint8
        avg_colored[:, :, 2] = np.clip(255 - avg_uint8, 0, 255)
        
        pil_layer = Image.fromarray(avg_colored)
        # Resize to standard size
        pil_layer_resized = pil_layer.resize((128, 128), Image.NEAREST)
        images_to_display.append((layer_name, pil_layer_resized, f'{spatial_size[0]}x{spatial_size[1]}x{num_channels}'))

# Display images one by one with labels
for layer_name, img, shape_info in images_to_display:
    print(f"\n🔹 {layer_name}: {shape_info}")
    display(img)

print("✓ Complete transformation pipeline visualized")
print(f"
Image journey:")
for layer_name, layer_output, spatial_size in conv_pool_layers:
    if layer_name == 'Input':
        print(f"  {layer_name}: {spatial_size[0]}x{spatial_size[1]}x3 (RGB image)")
    else:
        num_channels = layer_output.shape[-1]
        print(f"  {layer_name}: {spatial_size[0]}x{spatial_size[1]}x{num_channels} channels")

print("
💡 Observations:")
print("  • Spatial dimensions shrink: 32x32 → 16x16 → 8x8 → 4x4 (due to pooling)")
print("  • Number of channels increases: 3 → 32 → 64 → 64 (learning more features)")
print("  • Features become more abstract as we go deeper")
print("  • Early layers: edges, textures | Deeper layers: shapes, patterns")

# COMMAND ----------

# DBTITLE 1,📚 CNN Mathematics - Step by Step Explanation
# MAGIC %md
# MAGIC # 📚 CNN की Mathematics और Image Transformation
# MAGIC
# MAGIC ## यहां हम समझेंगे कि हर layer में क्या हो रहा है - Math के साथ!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔢 **Step 1: Input Image**
# MAGIC
# MAGIC **Input:** 32×32×3 image (32 height, 32 width, 3 color channels - Red, Green, Blue)
# MAGIC
# MAGIC **Example pixel values:**
# MAGIC ```
# MAGIC Red Channel [0-1]:    Green Channel:    Blue Channel:
# MAGIC 0.2 0.3 0.5 ...      0.1 0.4 0.6 ...   0.8 0.2 0.1 ...
# MAGIC 0.4 0.5 0.7 ...      0.3 0.5 0.7 ...   0.6 0.4 0.3 ...
# MAGIC ... (32×32 pixels)
# MAGIC ```
# MAGIC
# MAGIC **Total numbers:** 32 × 32 × 3 = **3,072 values**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⚙️ **Step 2: Convolution Operation (Conv2D Layer 1)**
# MAGIC
# MAGIC ### **Kya hota hai?**
# MAGIC Ek **filter/kernel** (3×3×3 size ka) image par slide करता है और **element-wise multiplication + sum** करता है।
# MAGIC
# MAGIC ### **Mathematics:**
# MAGIC
# MAGIC **Filter (3×3×3):**
# MAGIC ```
# MAGIC Filter weights (learnable):
# MAGIC Channel 1:        Channel 2:        Channel 3:
# MAGIC  0.1  -0.2  0.3    0.4  0.1  -0.3    -0.1  0.2  0.5
# MAGIC -0.4   0.5  0.2   -0.2  0.3   0.4     0.3 -0.1  0.2
# MAGIC  0.3  -0.1  0.4    0.1 -0.4   0.2     0.4  0.3 -0.2
# MAGIC ```
# MAGIC
# MAGIC **Operation at position (i, j):**
# MAGIC
# MAGIC ```
# MAGIC Output[i,j] = Σ Σ Σ (Image[i+m, j+n, c] × Filter[m, n, c]) + bias
# MAGIC               m n c
# MAGIC ```
# MAGIC
# MAGIC **Example calculation:**
# MAGIC ```
# MAGIC Image patch (3×3×3):     Filter (3×3×3):         Multiplication:
# MAGIC R: 0.2, 0.3, 0.5        × 0.1, -0.2, 0.3    =   0.02, -0.06, 0.15
# MAGIC    0.4, 0.5, 0.7          -0.4, 0.5, 0.2          -0.16, 0.25, 0.14
# MAGIC    0.1, 0.6, 0.3           0.3, -0.1, 0.4          0.03, -0.06, 0.12
# MAGIC
# MAGIC G: 0.1, 0.4, 0.6        × 0.4, 0.1, -0.3    =   0.04, 0.04, -0.18
# MAGIC    0.3, 0.5, 0.7          -0.2, 0.3, 0.4          -0.06, 0.15, 0.28
# MAGIC    0.2, 0.7, 0.4           0.1, -0.4, 0.2          0.02, -0.28, 0.08
# MAGIC
# MAGIC B: ... (same process)
# MAGIC
# MAGIC Sum of all = 0.02 + (-0.06) + 0.15 + ... + bias = 2.34 (example)
# MAGIC ```
# MAGIC
# MAGIC **Filter slide करता है:**
# MAGIC - Position (0,0) → calculation → Output[0,0]
# MAGIC - Position (0,1) → calculation → Output[0,1]
# MAGIC - ... और so on
# MAGIC
# MAGIC **Result:** 32×32×32 (32 filters लगाए, तो 32 feature maps बने)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔥 **Step 3: ReLU Activation**
# MAGIC
# MAGIC **Formula:** `ReLU(x) = max(0, x)`
# MAGIC
# MAGIC **Kya karta hai?**
# MAGIC - Negative values को 0 बना देता है
# MAGIC - Positive values same रहते हैं
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Before ReLU:  [-1.2, 2.3, -0.5, 3.1, 0.4]
# MAGIC After ReLU:   [ 0.0, 2.3,  0.0, 3.1, 0.4]
# MAGIC ```
# MAGIC
# MAGIC **Why?** Non-linearity introduce करता है, जिससे complex patterns सीख सकें।
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔽 **Step 4: MaxPooling (2×2)**
# MAGIC
# MAGIC **Kya hota hai?**
# MAGIC Har 2×2 window में से **maximum value** select करो।
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Input (4×4):              Output (2×2):
# MAGIC ┌──────────────┐          ┌──────┐
# MAGIC │ 1.2  2.3 │ 0.5  1.1 │      │ 2.3 │ 3.4 │
# MAGIC │ 0.8  1.5 │ 3.4  2.1 │      └──────┘
# MAGIC ├──────────────┤          │ 4.1 │ 2.9 │
# MAGIC │ 4.1  2.7 │ 2.9  1.8 │      └──────┘
# MAGIC │ 3.2  1.9 │ 0.7  2.2 │
# MAGIC └──────────────┘
# MAGIC
# MAGIC max(1.2, 2.3, 0.8, 1.5) = 2.3
# MAGIC max(0.5, 1.1, 3.4, 2.1) = 3.4
# MAGIC max(4.1, 2.7, 3.2, 1.9) = 4.1
# MAGIC max(2.9, 1.8, 0.7, 2.2) = 2.9
# MAGIC ```
# MAGIC
# MAGIC **Dimension change:** 32×32×32 → **16×16×32**
# MAGIC
# MAGIC **Why?**
# MAGIC - Spatial size reduce करता है (computation fast)
# MAGIC - Important features retain करता है
# MAGIC - Translation invariance (image थोड़ा shift हो तो भी same features)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔁 **Steps Repeat होते हैं:**
# MAGIC
# MAGIC | Layer | Input Size | Operation | Output Size | Parameters |
# MAGIC |-------|------------|-----------|-------------|------------|
# MAGIC | Input | 32×32×3 | - | 32×32×3 | 0 |
# MAGIC | Conv2D-1 | 32×32×3 | 32 filters (3×3) | 32×32×32 | 896 |
# MAGIC | ReLU | 32×32×32 | max(0,x) | 32×32×32 | 0 |
# MAGIC | MaxPool-1 | 32×32×32 | 2×2 pooling | **16×16×32** | 0 |
# MAGIC | Conv2D-2 | 16×16×32 | 64 filters (3×3) | 16×16×64 | 18,496 |
# MAGIC | ReLU | 16×16×64 | max(0,x) | 16×16×64 | 0 |
# MAGIC | MaxPool-2 | 16×16×64 | 2×2 pooling | **8×8×64** | 0 |
# MAGIC | Conv2D-3 | 8×8×64 | 64 filters (3×3) | 8×8×64 | 36,928 |
# MAGIC | ReLU | 8×8×64 | max(0,x) | 8×8×64 | 0 |
# MAGIC | MaxPool-3 | 8×8×64 | 2×2 pooling | **4×4×64** | 0 |
# MAGIC | Flatten | 4×4×64 | Reshape | **1024** | 0 |
# MAGIC | Dense | 1024 | Matrix multiply | 64 | 65,600 |
# MAGIC | Dropout | 64 | Random zeroing | 64 | 0 |
# MAGIC | Output | 64 | Matrix multiply + Softmax | **3** | 195 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🧮 **Final Dense Layer + Softmax**
# MAGIC
# MAGIC **Flatten:** 4×4×64 = 1024 values को एक line में करो
# MAGIC
# MAGIC **Dense Layer:**
# MAGIC ```
# MAGIC Output[i] = Σ (Input[j] × Weight[i,j]) + Bias[i]
# MAGIC             j
# MAGIC ```
# MAGIC
# MAGIC **Softmax:**
# MAGIC ```
# MAGIC Softmax(x[i]) = exp(x[i]) / Σ exp(x[j])
# MAGIC                              j
# MAGIC ```
# MAGIC
# MAGIC **Example:**
# MAGIC ```
# MAGIC Raw scores:    [2.3, 1.5, 0.8]  (for 3 classes)
# MAGIC
# MAGIC Softmax calculation:
# MAGIC exp(2.3) = 9.97
# MAGIC exp(1.5) = 4.48
# MAGIC exp(0.8) = 2.23
# MAGIC Sum = 16.68
# MAGIC
# MAGIC Probabilities:
# MAGIC Class 0 (airplane): 9.97/16.68 = 0.598 (59.8%)
# MAGIC Class 1 (automobile): 4.48/16.68 = 0.269 (26.9%)
# MAGIC Class 2 (bird): 2.23/16.68 = 0.134 (13.4%)
# MAGIC
# MAGIC Sum = 1.0 (100%)
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 **Summary: Image Journey**
# MAGIC
# MAGIC ```
# MAGIC 32×32×3 Image
# MAGIC     ↓ [Convolution: filters detect edges, colors]
# MAGIC 32×32×32 Feature Maps
# MAGIC     ↓ [Pooling: keep important features, reduce size]
# MAGIC 16×16×32
# MAGIC     ↓ [Convolution: detect shapes, textures]
# MAGIC 16×16×64
# MAGIC     ↓ [Pooling: more size reduction]
# MAGIC 8×8×64
# MAGIC     ↓ [Convolution: detect complex patterns]
# MAGIC 8×8×64
# MAGIC     ↓ [Pooling: final size reduction]
# MAGIC 4×4×64 = 1024 numbers
# MAGIC     ↓ [Flatten + Dense: combine features]
# MAGIC 64 neurons
# MAGIC     ↓ [Output + Softmax: class probabilities]
# MAGIC 3 probabilities (airplane, automobile, bird)
# MAGIC ```
# MAGIC
# MAGIC **Key Points:**
# MAGIC 1. **Spatial dimensions घटते हैं:** 32 → 16 → 8 → 4
# MAGIC 2. **Feature channels बढ़ते हैं:** 3 → 32 → 64 → 64
# MAGIC 3. **Early layers:** Simple features (edges, lines)
# MAGIC 4. **Deep layers:** Complex features (shapes, objects)
# MAGIC 5. **Final layer:** Class probabilities
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **अब नीचे के cells run करो to see actual mathematical operations! 👇**

# COMMAND ----------

# DBTITLE 1,Demo 1: Convolution Operation Mathematics
# 🔢 DEMO: Convolution Operation Step-by-Step

import numpy as np

print("="*70)
print("CONVOLUTION OPERATION - DETAILED MATHEMATICS")
print("="*70)

# Create a simple 6x6 image (single channel for simplicity)
image = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
])

# Create a 3x3 filter (edge detector)
filter_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

print("\n🖼️ INPUT IMAGE (6x6):")
print(image)
print(f"Shape: {image.shape}")

print("\n⚙️ FILTER/KERNEL (3x3 edge detector):")
print(filter_kernel)
print(f"Shape: {filter_kernel.shape}")

# Manual convolution
print("\n\n📊 CONVOLUTION PROCESS (Step by Step):")
print("-" * 70)

output = np.zeros((4, 4))  # Output will be 4x4

# Show first few calculations in detail
for i in range(2):  # Just show first 2 rows for clarity
    for j in range(2):  # Just show first 2 columns
        # Extract 3x3 region
        region = image[i:i+3, j:j+3]
        
        print(f"\nPosition ({i},{j}):")
        print(f"  Image Region (3x3):")
        print(f"  {region}")
        
        # Element-wise multiplication
        multiplication = region * filter_kernel
        print(f"\n  Filter × Region:")
        print(f"  {multiplication}")
        
        # Sum all values
        result = np.sum(multiplication)
        output[i, j] = result
        
        print(f"\n  Sum of all = {result}")
        print(f"  ⇒ Output[{i},{j}] = {result}")
        print("-" * 70)

# Complete convolution for full output
for i in range(4):
    for j in range(4):
        region = image[i:i+3, j:j+3]
        output[i, j] = np.sum(region * filter_kernel)

print("\n✅ FINAL OUTPUT (4x4):")
print(output)
print(f"Shape: {output.shape}")

# Apply ReLU
output_relu = np.maximum(0, output)
print("\n\n🔥 AFTER ReLU (negative → 0):")
print(output_relu)

# Display results as arrays (matplotlib unavailable due to numpy version)
print("\n📊 VISUALIZATIONS:")
print("\nOriginal Image (6x6):")
print(image)

print("\nFilter/Kernel (3x3 edge detector):")
print(filter_kernel)

print("\nOutput after Convolution (4x4):")
print(output)

print("\nOutput after ReLU (4x4):")
print(output_relu)

print("\n\n💡 KEY LEARNING:")
print("  1. Filter slides over image (convolution)")
print("  2. Element-wise multiply + sum at each position")
print("  3. Output size = (input_size - filter_size + 1)")
print("  4. In this example: 6 - 3 + 1 = 4")
print("  5. ReLU makes negative values zero")
print("  6. This filter detects edges in the image!")

# COMMAND ----------

# DBTITLE 1,Demo 2: MaxPooling Operation Mathematics
# 🔽 DEMO: MaxPooling Operation Step-by-Step

import numpy as np

print("="*70)
print("MAX POOLING OPERATION - DETAILED MATHEMATICS")
print("="*70)

# Create a 4x4 feature map (output from convolution)
feature_map = np.array([
    [1.2, 2.3, 0.5, 1.1],
    [0.8, 1.5, 3.4, 2.1],
    [4.1, 2.7, 2.9, 1.8],
    [3.2, 1.9, 0.7, 2.2]
])

print("\n📈 INPUT FEATURE MAP (4x4):")
print(feature_map)
print(f"Shape: {feature_map.shape}")

# 2x2 Max Pooling
print("\n\n🔍 MAX POOLING (2x2) - Step by Step:")
print("-" * 70)

pool_output = np.zeros((2, 2))

positions = [(0, 0), (0, 2), (2, 0), (2, 2)]
output_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for idx, ((i, j), (out_i, out_j)) in enumerate(zip(positions, output_positions)):
    # Extract 2x2 window
    window = feature_map[i:i+2, j:j+2]
    
    print(f"\nWindow at position ({i},{j}):")
    print(window)
    
    # Find maximum
    max_val = np.max(window)
    pool_output[out_i, out_j] = max_val
    
    print(f"Maximum value: {max_val}")
    print(f"⇒ Output[{out_i},{out_j}] = {max_val}")
    
    if idx < 3:
        print("-" * 70)

print("\n\n✅ FINAL POOLED OUTPUT (2x2):")
print(pool_output)
print(f"Shape: {pool_output.shape}")
print(f"\n📉 Size reduction: {feature_map.shape} → {pool_output.shape}")
print(f"Parameters reduced: {feature_map.size} → {pool_output.size} (75% reduction!)")

# Display as arrays
print("\n📊 VISUAL REPRESENTATION:")
print("\nInput Feature Map (4x4) with pooling windows marked:")
print("┌─────────┬─────────┐")
for i in range(4):
    row_str = "│ "
    for j in range(4):
        row_str += f"{feature_map[i,j]:.1f} "
        if j == 1:
            row_str += "│ "
    row_str += "│"
    print(row_str)
    if i == 1:
        print("├─────────┼─────────┤")
print("└─────────┴─────────┘")

print("\n     ↓ MaxPooling (2×2)")
print("\nPooled Output (2×2):")
print("┌─────────┐")
for i in range(2):
    row_str = "│ "
    for j in range(2):
        row_str += f"{pool_output[i,j]:.1f} "
    row_str += "│"
    print(row_str)
print("└─────────┘")

print("\n\n💡 KEY LEARNING:")
print("  1. MaxPooling reduces spatial dimensions (downsampling)")
print("  2. Keeps the STRONGEST feature in each window")
print("  3. Reduces computation and parameters")
print("  4. Makes model robust to small translations")
print("  5. No learnable parameters (just takes max)")
print("  6. Common pooling sizes: 2x2, 3x3")

print("\n🔢 MATH FORMULA:")
print("  Output[i,j] = max(Input[2i:2i+2, 2j:2j+2])")
print("  (for 2x2 pooling with stride 2)")

# COMMAND ----------

# DBTITLE 1,Demo 3: Softmax & Probability Calculation
# 🧮 DEMO: Softmax Operation - Converting Scores to Probabilities

import numpy as np

print("="*70)
print("SOFTMAX OPERATION - DETAILED MATHEMATICS")
print("="*70)

# Raw scores from the network (logits)
raw_scores = np.array([2.3, 1.5, 0.8])
class_names_demo = ['Airplane', 'Automobile', 'Bird']

print("\n📊 RAW SCORES (Logits) from Dense Layer:")
for i, (score, name) in enumerate(zip(raw_scores, class_names_demo)):
    print(f"  Class {i} ({name}): {score:.2f}")

print("\n\n📝 SOFTMAX CALCULATION - Step by Step:")
print("-" * 70)

print("\nFormula: Softmax(x_i) = exp(x_i) / Σ(exp(x_j))")
print("                              j")

# Step 1: Calculate exponentials
print("\n\nStep 1: Calculate exp() for each score")
exp_scores = np.exp(raw_scores)
for i, (score, exp_score, name) in enumerate(zip(raw_scores, exp_scores, class_names_demo)):
    print(f"  exp({score:.2f}) = {exp_score:.4f}  ({name})")

# Step 2: Sum of exponentials
exp_sum = np.sum(exp_scores)
print(f"\n\nStep 2: Sum of all exponentials")
print(f"  {exp_scores[0]:.4f} + {exp_scores[1]:.4f} + {exp_scores[2]:.4f} = {exp_sum:.4f}")

# Step 3: Divide each by sum
print(f"\n\nStep 3: Divide each exp by the sum")
probabilities = exp_scores / exp_sum

for i, (exp_score, prob, name) in enumerate(zip(exp_scores, probabilities, class_names_demo)):
    print(f"  {exp_score:.4f} / {exp_sum:.4f} = {prob:.4f} = {prob*100:.2f}%  ({name})")

print("\n" + "-" * 70)
print("\n✅ FINAL PROBABILITIES:")
for i, (prob, name) in enumerate(zip(probabilities, class_names_demo)):
    print(f"  {name}: {prob:.4f} ({prob*100:.2f}%)")

print(f"\n📊 Sum check: {np.sum(probabilities):.6f} (should be exactly 1.0)")

# Prediction
predicted_class = np.argmax(probabilities)
print(f"\n🎯 PREDICTION: Class {predicted_class} ({class_names_demo[predicted_class]})")
print(f"   Confidence: {probabilities[predicted_class]*100:.2f}%")

# Visual representation using ASCII bars
print("\n\n📊 VISUAL COMPARISON:")
print("\n1. Raw Scores:")
for i, (score, name) in enumerate(zip(raw_scores, class_names_demo)):
    bar = '█' * int(score * 10)
    print(f"  {name:12s}: {score:.2f} {bar}")

print("\n2. After exp():")
for i, (exp_score, name) in enumerate(zip(exp_scores, class_names_demo)):
    bar = '█' * int(exp_score / 2)
    print(f"  {name:12s}: {exp_score:.2f} {bar}")

print("\n3. Probabilities (Softmax):")
for i, (prob, name) in enumerate(zip(probabilities, class_names_demo)):
    bar = '█' * int(prob * 50)
    marker = ' ← WINNER!' if i == predicted_class else ''
    print(f"  {name:12s}: {prob*100:5.1f}% {bar}{marker}")

print("\n\n💡 KEY LEARNING:")
print("  1. Softmax converts raw scores to probabilities")
print("  2. All probabilities sum to exactly 1.0 (100%)")
print("  3. Higher score → higher probability")
print("  4. Exponential makes differences more pronounced")
print("  5. Used for multi-class classification")
print("  6. Prediction = class with highest probability")

print("\n\n🔥 WHY exp() ?")
print("  - Amplifies differences between scores")
print("  - Always positive (probabilities can't be negative)")
print("  - Makes gradient computation easier during training")
print("  - Example: score difference of 1.5 becomes exp ratio of ~4.5x")

# Show comparison
print("\n\n🔍 COMPARISON - What if we just normalized without exp?")
simple_norm = raw_scores / np.sum(raw_scores)
print("\nSimple normalization (without exp):")
for i, (prob, name) in enumerate(zip(simple_norm, class_names_demo)):
    print(f"  {name}: {prob:.4f} ({prob*100:.2f}%)")

print("\nWith Softmax (using exp):")
for i, (prob, name) in enumerate(zip(probabilities, class_names_demo)):
    print(f"  {name}: {prob:.4f} ({prob*100:.2f}%)")

print("\n⇒ Notice: Softmax makes the winner more confident!")
print(f"   Simple: {simple_norm[0]*100:.1f}% vs Softmax: {probabilities[0]*100:.1f}%")

# COMMAND ----------

# DBTITLE 1,Demo 4: Complete Network Flow with Actual Data
# 🌊 DEMO: Complete Network Flow - Real Image through CNN

import numpy as np
# Using already loaded model and feature_extraction_model from previous cells

print("="*70)
print("COMPLETE CNN FLOW - TRACKING ONE IMAGE THROUGH ALL LAYERS")
print("="*70)

# Get a real test image
if 'X_test' in dir() and len(X_test) > 0:
    test_idx = 0
    sample_image = X_test[test_idx]
    true_class = y_test[test_idx][0]
    
    print(f"\n🖼️ Selected Image: {class_names[true_class]}")
    print(f"   Index: {test_idx}")
    print(f"   Image is a real 32x32 color photo from CIFAR-10 dataset!")
    
    print("\n" + "="*70)
    print("LAYER-BY-LAYER TRANSFORMATION (with numbers!)")
    print("="*70)
    
    # Track through layers
    print(f"\n🔢 LAYER 0: INPUT")
    print(f"   Shape: {sample_image.shape}")
    print(f"   Total values: {sample_image.size}")
    print(f"   Value range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    print(f"   Sample pixel (10,10): R={sample_image[10,10,0]:.3f}, G={sample_image[10,10,1]:.3f}, B={sample_image[10,10,2]:.3f}")
    
    # Get activations from all layers
    if 'feature_extraction_model' in dir():
        input_batch = sample_image.reshape(1, 32, 32, 3)
        activations = feature_extraction_model.predict(input_batch, verbose=0)
        
        layer_names = [layer.name for layer in model.layers[1:]]
        
        # Display each layer's output
        for idx, (activation, layer_name) in enumerate(zip(activations, layer_names)):
            activation_squeezed = activation[0]  # Remove batch dimension
            
            print(f"\n⚙️ LAYER {idx+1}: {layer_name.upper()}")
            print(f"   Input shape: {activations[idx-1][0].shape if idx > 0 else sample_image.shape}")
            print(f"   Output shape: {activation_squeezed.shape}")
            print(f"   Total values: {activation_squeezed.size}")
            print(f"   Value range: [{activation_squeezed.min():.3f}, {activation_squeezed.max():.3f}]")
            
            if len(activation_squeezed.shape) == 3:  # Conv or Pooling layer
                print(f"   Spatial size: {activation_squeezed.shape[0]}x{activation_squeezed.shape[1]}")
                print(f"   Number of channels/filters: {activation_squeezed.shape[2]}")
                
                # Show sample values from first filter
                print(f"   Sample values from filter 0:")
                sample_vals = activation_squeezed[:3, :3, 0]
                for row in sample_vals:
                    print(f"     {' '.join([f'{v:6.2f}' for v in row])}")
                
                # Check for dead neurons (all zeros)
                zero_channels = np.sum(np.max(activation_squeezed, axis=(0,1)) == 0)
                if zero_channels > 0:
                    print(f"   ⚠️  {zero_channels} dead filters (all zeros)")
                    
            elif len(activation_squeezed.shape) == 1:  # Flatten or Dense layer
                print(f"   Vector length: {activation_squeezed.shape[0]}")
                print(f"   Non-zero values: {np.count_nonzero(activation_squeezed)}")
                print(f"   Sample values: [{', '.join([f'{v:.3f}' for v in activation_squeezed[:5]])}...]")
            
            # Show size reduction
            if idx > 0:
                prev_size = activations[idx-1][0].size
                curr_size = activation_squeezed.size
                reduction = (1 - curr_size/prev_size) * 100
                if reduction > 0:
                    print(f"   📉 Size reduction: {prev_size} → {curr_size} ({reduction:.1f}% smaller)")
                elif reduction < 0:
                    print(f"   📈 Size increase: {prev_size} → {curr_size} ({-reduction:.1f}% larger)")
        
        # Final prediction
        print("\n" + "="*70)
        print("🎯 FINAL PREDICTION")
        print("="*70)
        
        predictions = model.predict(input_batch, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        
        print(f"\n📊 Class Probabilities:")
        for i, (prob, name) in enumerate(zip(predictions, class_names)):
            bar = '█' * int(prob * 50)
            marker = ' ← PREDICTION' if i == predicted_class else ''
            correct = ' ✓ CORRECT' if i == true_class else ''
            print(f"   {name:12s}: {prob:.4f} ({prob*100:5.2f}%) {bar}{marker}{correct}")
        
        print(f"\n✅ Predicted: {class_names[predicted_class]} ({predictions[predicted_class]*100:.2f}% confidence)")
        print(f"✅ Actual: {class_names[true_class]}")
        
        if predicted_class == true_class:
            print("\n🎉 CORRECT PREDICTION!")
        else:
            print("\n❌ INCORRECT PREDICTION")
        
        # Summary
        print("\n" + "="*70)
        print("📊 TRANSFORMATION SUMMARY")
        print("="*70)
        print(f"\n  Input:  {sample_image.shape} = {sample_image.size:,} values")
        print(f"    ↓")
        print(f"  After Conv+Pool layers: {activations[5][0].shape} = {activations[5][0].size:,} values")
        print(f"    ↓")
        print(f"  After Flatten: {activations[6][0].shape} = {activations[6][0].size:,} values")
        print(f"    ↓")
        print(f"  After Dense: {activations[7][0].shape} = {activations[7][0].size:,} values")
        print(f"    ↓")
        print(f"  Final Output: {predictions.shape} = {predictions.size} probabilities")
        
        print(f"\n  Total reduction: {sample_image.size:,} → {predictions.size} values")
        print(f"  That's a {(1 - predictions.size/sample_image.size)*100:.1f}% reduction!")
        
    else:
        print("\n⚠️  Run the 'Create Feature Extraction Model' cell first!")
        print("   (Cell 18 in the notebook)")
else:
    print("\n⚠️  No test data available. Run the training cells first!")
    print("   (Cells 5-12 in the notebook)")

print("\n\n💡 KEY OBSERVATIONS:")
print("  1. Spatial dimensions SHRINK: 32x32 → 16x16 → 8x8 → 4x4")
print("  2. Feature channels GROW: 3 → 32 → 64 → 64")
print("  3. After flatten: Long vector (1024 values)")
print("  4. Dense layers: Extract high-level features")
print("  5. Final layer: 3 probabilities (one per class)")
print("  6. Network learns: pixels → edges → shapes → objects → classes")