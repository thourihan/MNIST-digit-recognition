import tensorflow as tf
import gradio as gr

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=6)

def classify(input):
  normalized_input = input / 255.0
  prediction = model.predict(normalized_input.reshape(1, 28, 28, 1)).tolist()[0]
  return {str(i): prediction[i] for i in range(10)}

label = gr.outputs.Label(num_top_classes=10)
interface = gr.Interface(fn=classify, inputs="sketchpad", outputs=label, live=True)
interface.launch()
