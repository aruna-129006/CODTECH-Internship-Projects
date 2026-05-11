import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\aruna\Downloads\dataset\train",
    image_size=(128, 128),
    batch_size=16
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    r"C:\Users\aruna\Downloads\dataset\test",
    image_size=(128, 128),
    batch_size=16
)

# Normalize
train_data = train_data.map(lambda x, y: (x/255.0, y))
test_data = test_data.map(lambda x, y: (x/255.0, y))

# Build model
model = models.Sequential([
    layers.Input(shape=(128,128,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# Plot graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Test'])
plt.show()

# Save model
model.save("model.keras")

print("Model saved successfully")
