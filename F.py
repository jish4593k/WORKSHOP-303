import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image, ImageTk

# Define a simple CNN model using PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * 112 * 112, 2)  # Assuming input image size is 224x224

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the trained Keras model for image processing
keras_model = tf.keras.models.load_model('your_keras_model.h5')

def process_image_with_pytorch(image_path):
    # Process image using PyTorch (replace with your actual image processing logic)
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    # Use a simple CNN model
    pytorch_model = SimpleCNN()
    output = pytorch_model(img_tensor)

    return output

def process_image_with_keras(image_path):
    # Process image using Keras (replace with your actual image processing logic)
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Adjust size based on your model input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values

    # Use the trained Keras model
    output = keras_model.predict(img_array)

    return output

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("COVID-19 Data Processing App")

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.label = tk.Label(self.frame, text="Select an image:")
        self.label.pack()

        self.image_path = None

        self.browse_button = tk.Button(self.frame, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        self.process_button = tk.Button(self.frame, text="Process Image", command=self.process_image)
        self.process_button.pack()

    def browse_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])

    def process_image(self):
        if self.image_path:
            # Process image with PyTorch
            pytorch_output = process_image_with_pytorch(self.image_path)
            print("PyTorch Output:", pytorch_output)

            # Process image with Keras
            keras_output = process_image_with_keras(self.image_path)
            print("Keras Output:", keras_output)

            # Display the processed image
            processed_img = Image.open(self.image_path)
            processed_img.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
