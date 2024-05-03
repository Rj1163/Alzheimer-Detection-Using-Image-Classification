import pathlib
import numpy as np
import tensorflow as tf

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Define file path and data directory
file_path = "C:\\Users\\Riddhi\\Downloads\\ALZ"
data_dir = pathlib.Path(file_path)

# Define constants
SEED = 1
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
INPUT_SHAPE = IMG_SIZE + (3,)

# Prepare train and validation datasets
X_train = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training'
)

X_validation = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation'
)

# Define class labels
class_labels = ['Non-demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(scale=1./255, input_shape=INPUT_SHAPE),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train,
    validation_data=X_validation,
    epochs=10
)

# Save the model
model.save('model.h5')

# Extract true labels from validation dataset
true_labels = np.concatenate([y for x, y in X_validation], axis=0)

# Generate predictions
predictions = np.argmax(model.predict(X_validation), axis=-1)

# Generate confusion matrix
conf_matrix = tf.math.confusion_matrix(labels=true_labels, predictions=predictions).numpy()

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Calculate accuracy
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print("Accuracy:", accuracy)

# Print model summary
print("Model Summary:")
model.summary()

# Print predicted class labels
print("Predicted Class Labels:")
for i, pred in enumerate(predictions):
    print(f"Predicted: {class_labels[pred]}, True Label: {class_labels[true_labels[i]]}")
