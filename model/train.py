import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
from cnn_model import create_cnn_model
from tensorflow.keras.optimizers import Adam

# ================= CONFIG =================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 2
CLASS_NAMES = ['dangerous', 'safe']

DATASET_PATH = '../dataset'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
TEST_PATH = os.path.join(DATASET_PATH, 'test')
MODEL_SAVE_PATH = './saved_models'

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ================= DATA GENERATORS =================
def create_data_generators():
    # Light augmentation to avoid exploding val_loss
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# ================= PLOT HISTORY =================
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_history.png'))
    plt.show()
    print("✅ Training history saved!")

# ================= EVALUATE =================
def evaluate_model(model, test_generator):
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype("int32").reshape(-1)
    y_true = test_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'))
    plt.show()
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\n📊 Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"📊 Test Loss: {test_loss:.4f}")

# ================= TRAIN =================
def train_model():
    print("="*50)
    print("DANGEROUS OBJECT DETECTION - CNN TRAINING")
    print("="*50)
    
    train_gen, val_gen, test_gen = create_data_generators()
    print(f"Train samples: {train_gen.samples}, Val samples: {val_gen.samples}, Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # ================= CLASS WEIGHTS =================
    # Automatic class weight based on train samples
    y_train = train_gen.classes
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weights[i] for i in range(len(weights))}
    print(f"Class weights: {class_weight_dict}")

    # ================= MODEL =================
    model = create_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE,3))
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print("✅ Model created and compiled!")

    # ================= CALLBACKS =================
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ================= TRAIN =================
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_model.h5')
    model.save(final_model_path)
    print(f"✅ Final model saved at: {final_model_path}")

    plot_training_history(history)
    evaluate_model(model, test_gen)

    print("✅ TRAINING COMPLETED")

if __name__ == "__main__":
    train_model()
