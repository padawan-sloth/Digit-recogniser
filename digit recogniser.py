import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load the data
train_data = pd.read_csv('C:/Users/Fierce/Desktop/Personal Projects/Personal-Projects/Digit Recogniser/train.csv')
test_data = pd.read_csv('C:/Users/Fierce/Desktop/Personal Projects/Personal-Projects/Digit Recogniser/test.csv')

# Normalize the pixel values
X_train = train_data.drop('label', axis=1) / 255.0
y_train = train_data['label']
X_test = test_data / 255.0

# Reshape the data to fit the model input requirements
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

# Optionally, save the model
model.save('mnist_model.h5')

# Predict on the test data
test_predictions = model.predict(X_test)

# Since the output will be in one-hot format, convert predictions to labels
predicted_labels = np.argmax(test_predictions, axis=1)

# Create a DataFrame for the submission
submission_df = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

# Save the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Save the DataFrame to a CSV file at the specified path
submission_df.to_csv('C:/Users/Fierce/Desktop/Personal Projects/Personal-Projects/Digit Recogniser/submission.csv', index=False)
