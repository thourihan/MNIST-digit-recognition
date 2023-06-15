# MNIST Digit Recognition

This repository contains an implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras for recognizing handwritten digits. The model is trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of images of handwritten digits. A web interface for the model is created using Gradio.

## Project Structure

The repository contains the following files:

- `main.py`: The main Python script that includes the implementation of the CNN and the Gradio interface.
- `requirements.txt`: A list of Python dependencies required to run the script.

## Installation

1. Clone the repository:
```
git clone https://github.com/thourihan/MNIST-digit-recognition.git
cd MNIST-digit-recognition
```
2. Create a virtual environment and activate it (Optional but recommended):
```
python3 -m venv env
source env/bin/activate # On Windows use env\Scripts\activate
```
3. Install the required packages:
```
pip install -r requirements.txt
```
## Usage

Run the `main.py` script:
```
python main.py
```
This will start the Gradio interface where you can draw a digit and see the model's prediction.

## About the Model

The CNN model architecture consists of the following layers:

- Input reshaping to 28x28x1
- Convolutional layer with 32 filters of size 3x3 and ReLU activation
- 2x2 Max pooling layer
- Flatten layer
- Dense layer with 128 units and ReLU activation
- Output layer with 10 units (for the 10 digit classes) and softmax activation

The model is compiled with Adam optimizer and sparse categorical cross-entropy loss function. It is trained for 6 epochs.
