# Based on:
# https://www.tensorflow.org/lite/convert
# https://www.tensorflow.org/lite/guide/inference

import tensorflow as tf
import numpy as np

saved_model_path = 'src/saved_model/mnist_reusability'
tflite_model_path = 'src/tflite_models/model.tflite'

def convert_model():

	model = tf.keras.models.load_model(saved_model_path)

	# print(model.summary())

	# Convert the model
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()

	# Save the model.
	with open(tflite_model_path, 'wb') as f:
		f.write(tflite_model)


def predict():
	interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
	interpreter.allocate_tensors()

	# view input and output details
	input_details = interpreter.get_input_details()
	print(input_details)
	output_details = interpreter.get_output_details()
	print(output_details)

	# get a test image
	(train_images, train_labels), (test_images, test_labels) =  tf.keras.datasets.fashion_mnist.load_data()
	train_images = train_images / 255.0
	test_images = test_images / 255.0
	img = test_images[1]
	img = (np.expand_dims(img,0))

	class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

	# set input tensor
	tensor_index = interpreter.get_input_details()[0]['index']
	interpreter.tensor(tensor_index)()[0] = img

	# run inference
	interpreter.invoke()

	# get output tensor
	output_data = interpreter.get_tensor(output_details[0]['index'])
	print(output_data)

	# translate result to predicted class name
	predicted_label_index = np.argmax(output_data[0])
	print(predicted_label_index)
	print(class_names[predicted_label_index])

convert_model()
# predict()
