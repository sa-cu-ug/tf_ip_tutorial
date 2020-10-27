# Based on:
# https://www.tensorflow.org/tutorials/keras/classification
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
# https://www.tensorflow.org/tutorials/keras/save_and_load

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display

class Inferer():

	def __init__(self, load_model=False, load_checkpoint=False):

		print('Tensorflow version:')
		print(tf.__version__)

		(self. train_images, self.train_labels), (self.test_images, self.test_labels) =  tf.keras.datasets.fashion_mnist.load_data()

		print('Training images shape:')
		print(self.train_images.shape)

		self.input_shape = (self.train_images.shape[1], self.train_images.shape[2])
		print('Training images shape without batch size:')
		print(self.input_shape)

		self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

		# view sample and recognize the value range [0, 255]
		# self.show_sample(0)

		# scale to [0, 1]
		self.train_images = self.train_images / 255.0
		self.test_images = self.test_images / 255.0

		# view sample and recognize the value scaled range [0, 1]
		# self.show_sample(0)

		# view samples with labels to get an extended impression of the dataset
		# self.show_samples_with_labels()

		self.checkpoint_path = "tutorial_image_classification/checkpoints/cp-{epoch:04d}.ckpt"
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

		# Create a callback that saves the model's weights
		self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
												save_weights_only=True,
												verbose=1,
												period=5) # every five epochs

		self.saved_model_path = 'tutorial_image_classification/saved_model/mnist_reusability'

		if load_model == True:
			try:
				# load previously saved model
				self.model = tf.keras.models.load_model(self.saved_model_path)
				print('model loaded:')
				print(self.model.summary())
			except Exception as e:
				exit(e)

		else:
			# create simple sequential model with 3 layers
			self.model = self.create_model(load_checkpoint)

		# compile with chosen optimizer and loss function
		self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

		# training options:
		self.epochs = 10


	def show_sample(self, index):

		display.clear_output(wait=True)
		plt.figure()
		plt.imshow(self.train_images[index])
		plt.colorbar()
		plt.grid(False)
		plt.show()

	def show_samples_with_labels(self):
		display.clear_output(wait=True)
		plt.figure(figsize=(10,10))
		for i in range(25):
			plt.subplot(5,5,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(self.train_images[i], cmap=plt.cm.binary)
			plt.xlabel(self.class_names[self.train_labels[i]])
		plt.show()

	def create_model(self, load_checkpoint):
		model = tf.keras.Sequential([
			tf.keras.layers.Flatten(input_shape=self.input_shape),
			tf.keras.layers.Dense(128, activation='relu'),
			tf.keras.layers.Dense(10)
		])

		# Display the model's architecture
		# print(model.summary())

		if load_checkpoint == True:
			# load weights, if checkpoint data exists
			latest = tf.train.latest_checkpoint(self.checkpoint_dir)
			if latest != None:
				print('Checkpoint found:')
				print(latest)
				model.load_weights(latest)
				print('Checkpoint loaded')
				return model
			else:
				print('No checkpoint found')

		# model.save_weights(self.checkpoint_path.format(epoch=0))

		return model

	def train(self, save_checkpoints):

		if save_checkpoints:
			self.model.fit(
				self.train_images,
				self.train_labels,
				epochs=self.epochs,
				validation_data=(self.test_images, self.test_labels), # use test data for validation
				callbacks=[self.cp_callback]) # Pass callback to training
		else:
			self.model.fit(
				self.train_images,
				self.train_labels,
				epochs=self.epochs,
				validation_data=(self.test_images, self.test_labels)) # use test data for validation # Pass callback to training

	def test(self):
		test_loss, test_acc = self.model.evaluate(self.test_images,  self.test_labels, verbose=2)
		print('\nTest accuracy:', test_acc)

	def create_propability_model(self):
		self.probability_model = tf.keras.Sequential([self.model,
                                         tf.keras.layers.Softmax()])

	def show_predictions(self):

		predictions = self.probability_model.predict(self.test_images)

		num_rows = 5
		num_cols = 3
		num_images = num_rows*num_cols
		plt.figure(figsize=(2*2*num_cols, 2*num_rows))

		for i in range(num_images):
			plt.subplot(num_rows, 2*num_cols, 2*i+1)
			self.plot_image(i, predictions[i], self.test_labels, self.test_images)
			plt.subplot(num_rows, 2*num_cols, 2*i+2)
			self.plot_value_array(i, predictions[i], self.test_labels)

		plt.tight_layout()
		plt.show()

	def plot_image(self, i, predictions_array, true_label, img):
		true_label, img = true_label[i], img[i]
		plt.grid(False)
		plt.xticks([])
		plt.yticks([])

		plt.imshow(img, cmap=plt.cm.binary)

		predicted_label = np.argmax(predictions_array)
		if predicted_label == true_label:
			color = 'blue'
		else:
			color = 'red'

		plt.xlabel("{} {:2.0f}% ({})".format(self.class_names[predicted_label],
										100*np.max(predictions_array),
										self.class_names[true_label]),
										color=color)

	def plot_value_array(self, i, predictions_array, true_label):
		true_label = true_label[i]
		plt.grid(False)
		plt.xticks(range(10))
		plt.yticks([])
		thisplot = plt.bar(range(10), predictions_array, color="#777777")
		plt.ylim([0, 1])
		predicted_label = np.argmax(predictions_array)

		thisplot[predicted_label].set_color('red')
		thisplot[true_label].set_color('blue')

	def test_single_image(self):
		img = self.test_images[1]

		# recognize the image shape is withouth the batch size
		print(img.shape)

		# expand dimensions to get a batch size of 1 for a single image
		img = (np.expand_dims(img,0))

		# recognize the shape is now expanded with batch size
		print(img.shape)

		self.show_sample(1)

		predictions_single = self.probability_model.predict(img)

		# view predictions array
		print(predictions_single)

		display.clear_output(wait=True)
		plt.figure(figsize=(10,10))
		self.plot_value_array(1, predictions_single[0], self.test_labels)
		_ = plt.xticks(range(10), self.class_names, rotation=45)
		plt.show()

		# index of the max prediction (=> index of predicted label)
		print(np.argmax(predictions_single[0]))

	def save_model(self):
		self.model.save(self.saved_model_path)

if __name__ == "__main__":

	inferer = Inferer(True, False)

	inferer.train(False)

	inferer.save_model()

	# inferer.test()

	# inferer.create_propability_model()

	# inferer.show_predictions()

	# inferer.test_single_image()

	# inferer.save_model()

