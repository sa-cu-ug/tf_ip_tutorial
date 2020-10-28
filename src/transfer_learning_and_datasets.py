# Based on:
# https://www.tensorflow.org/tutorials/images/transfer_learning
# https://www.tensorflow.org/datasets/overview

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as display
import tensorflow_datasets as tfds

class Inferer():

	def __init__(self):

		# training options:
		self.epochs = 10
		self.batch_size = 32

		builder = tfds.builder('coil100')
		# 1. Create the tfrecord files (no-op if already exists)
		builder.download_and_prepare()
		# 2. Load the `tf.data.Dataset`
		self.dataset = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)

		self.input_shape = (128, 128, 3)

		# self.show_sample()
		self.show_samples_with_label_indices()

		# 80% train, 20% validation
		train_batches = tf.data.experimental.cardinality(self.dataset)
		self.validation_dataset = self.dataset.take(train_batches // 5)
		self.train_dataset = self.dataset.skip(train_batches // 5)

		AUTOTUNE = tf.data.experimental.AUTOTUNE

		self.train_dataset = self.train_dataset.cache()
		self.train_dataset = self.train_dataset.batch(self.batch_size)
		self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
		self.validation_dataset = self.validation_dataset.batch(self.batch_size)
		self.validation_dataset = self.validation_dataset.cache()
		self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)

		self.model = self.create_transfer_model()

		base_learning_rate = 0.0001
		self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
					loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
					metrics=['accuracy'])
		# view summary of the chained model
		print(self.model.summary())

	def show_sample(self):

		ds = self.dataset.take(1)

		display.clear_output(wait=True)
		plt.figure()

		for example in ds:
			plt.imshow(example["image"])
		plt.colorbar()
		plt.grid(False)
		plt.show()

	def show_samples_with_label_indices(self):

		ds = self.dataset.take(25)

		display.clear_output(wait=True)
		plt.figure(figsize=(10,10))

		i = 0
		for image, label in ds:
			plt.subplot(5,5,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			plt.imshow(image, cmap=plt.cm.binary)
			plt.xlabel(label.numpy())
			i += 1
		plt.show()

	def create_transfer_model(self):

		# load pretrained model without classification head
		pretrained_model = tf.keras.applications.MobileNetV2(input_shape=self.input_shape,
                                               include_top=False,
                                               weights='imagenet')

		pretrained_model.trainable = False

		# define preprocess layer
		preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

		# add classification head
		image_batch, label_batch = next(iter(self.train_dataset))
		feature_batch = pretrained_model(image_batch)

		global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
		feature_batch_average = global_average_layer(feature_batch)

		# add prediction layer
		prediction_layer = tf.keras.layers.Dense(1)
		prediction_batch = prediction_layer(feature_batch_average)

		# chain together
		inputs = tf.keras.Input(shape=self.input_shape)
		x = preprocess_input(inputs)
		x = pretrained_model(x, training=False)
		x = global_average_layer(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		outputs = prediction_layer(x)
		model = tf.keras.Model(inputs, outputs)

		return model


	def train(self):
		history = self.model.fit(self.train_dataset,
                    epochs=self.epochs,
                    validation_data=self.validation_dataset)



if __name__ == "__main__":

	inferer = Inferer()

	inferer.train()

