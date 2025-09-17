import tensorflow as tf
import os

# Define dataset paths
DATASET_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data'
TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
VAL_PATH = os.path.join(DATASET_PATH, 'val')
TEST_PATH = os.path.join(DATASET_PATH, 'test')

# Define data augmentation
data_augmentation = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load training data with data augmentation
train_data = data_augmentation.flow_from_directory(
    TRAIN_PATH,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data without data augmentation, but with rescaling
val_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    VAL_PATH,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

# Load test data without data augmentation, but with rescaling
test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_PATH,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)

print("Data Loaded Successfully.")
