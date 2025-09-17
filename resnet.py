import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define dataset paths
TRAIN_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
VAL_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val'
TEST_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\test'

try:
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU available")
        tf.config.set_visible_devices(gpus[0], 'GPU')
    else:
        print("GPU not available")

    # Load ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Freeze base layers
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(11, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

    # Train the model
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save the model
    model.save(r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\best\resnet50_fish_classifier.h5')

except Exception as e:
    print(f"An error occurred: {e}")