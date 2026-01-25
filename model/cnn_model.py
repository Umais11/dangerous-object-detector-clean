"""
Optimized CNN Model for Dangerous Object Detection
Binary Classification: dangerous vs safe
Input Size: 128x128x3
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(128, 128, 3)):
    
    model = Sequential([
        # ===== Block 1 =====
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.2),

        # ===== Block 2 =====
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.3),

        # ===== Block 3 =====
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.3),

        # ===== Block 4 =====
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.4),

        # ===== Global Pooling =====
        GlobalAveragePooling2D(),

        # ===== Fully Connected =====
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # ===== Output Layer =====
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_model_summary():
    """
    Displays model architecture summary
    """
    model = create_cnn_model()
    return model.summary()


if __name__ == "__main__":
    print("Creating Optimized CNN Model...")
    model = create_cnn_model()
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    print("\n" + "="*50)
    print(f"Total Parameters: {model.count_params():,}")
    print("="*50)
