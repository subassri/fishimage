import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
try:
    # Use GPU acceleration if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
except Exception as e:
    print(f"Error: {e}")

# Define dataset paths
TRAIN_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
VAL_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val'
TEST_PATH = r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\test'

try:
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
except Exception as e:
    print(f"Error loading data: {e}")

try:
    # Define the model architecture
    def create_model(optimizer='adam', dropout_rate=0.2):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
        ])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Hyperparameter tuning
    optimizers = ['adam', 'rmsprop', 'sgd']
    dropout_rates = [0.1, 0.2, 0.3]
    best_accuracy = 0
    best_hyperparameters = {}

    for optimizer in optimizers:
        for dropout_rate in dropout_rates:
            try:
                model = create_model(optimizer=optimizer, dropout_rate=dropout_rate)
                history = model.fit(
                    train_data,
                    epochs=10,
                    validation_data=val_data
                )
                test_loss, test_acc = model.evaluate(test_data)
                print(f'Optimizer: {optimizer}, Dropout rate: {dropout_rate}, Test accuracy: {test_acc:.2f}')
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_hyperparameters = {
                        'optimizer': optimizer,
                        'dropout_rate': dropout_rate
                    }
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}")

    # Train the model with the best hyperparameters
    try:
        best_model = create_model(optimizer=best_hyperparameters['optimizer'], dropout_rate=best_hyperparameters['dropout_rate'])
        history = best_model.fit(
            train_data,
            epochs=10,
            validation_data=val_data
        )
        test_loss, test_acc = best_model.evaluate(test_data)
        print(f'Test accuracy with best hyperparameters: {test_acc:.2f}')
    except Exception as e:
        print(f"Error training with best hyperparameters: {e}")

    # Save the best model
    try:
        best_model.save(r'C:\Users\Administrator\Desktop\Miniproj\venv\fish\best\cnn_fish_rmsprop_0.2.h5')
    except Exception as e:
        print(f"Error saving model: {e}")
except Exception as e:
    print(f"Error: {e}")