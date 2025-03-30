import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Constants (these should match the values used during training)
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')


def load_image():
    """Load an image using the file dialog and update the GUI."""
    file_path = filedialog.askopenfilename()
    
    if file_path:
        # Open the image and display it
        img = Image.open(file_path)
        img = img.resize((200, 200))  # Resize for display
        img = ImageTk.PhotoImage(img)

        # Update the image on the GUI
        panel.config(image=img)
        panel.image = img
        
        # Process the image for prediction
        predict_image(file_path)


def predict_image(file_path):
    """Preprocess the image and make a prediction using the trained model."""
    # Read the image using OpenCV
    img = cv2.imread(file_path)

    # Check if image is valid
    if img is None:
        messagebox.showerror("Error", "Invalid image file")
        return

    # Resize and normalize the image
    img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img_resized = img_resized / 255.0  # Normalize image pixels to [0, 1]

    # Reshape to match the model input (batch_size, IMG_WIDTH, IMG_HEIGHT, 3)
    img_array = np.expand_dims(img_resized, axis=0)

    # Make the prediction
    predictions = model.predict(img_array)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Display the predicted class
    result_label.config(text=f"Predicted Class: {predicted_class}")


def create_gui():
    """Create the GUI with Tkinter."""
    window = tk.Tk()
    window.title("Traffic Sign Classifier")

    # Create an upload button to select an image
    upload_button = tk.Button(window, text="Upload Traffic Sign Image",
                              command=load_image)
    upload_button.pack(pady=20)

    # Create a label to display the image
    global panel
    panel = tk.Label(window)
    panel.pack(pady=20)

    # Create a label to show the prediction result
    global result_label
    result_label = tk.Label(window, text="Predicted Class: None", 
                            font=("Arial", 14))
    result_label.pack(pady=10)

    # Run the Tkinter event loop
    window.mainloop()


if __name__ == "__main__":
    create_gui()
