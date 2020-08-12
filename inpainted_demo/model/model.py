import tensorflow.keras as keras
from tensorflow.keras import layers

def ResidualBlock(x):
	filters = 64
	kernel_size = 3
	momentum = 0.8
	padding = 'same'
	strides = 1


	res = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
						strides=strides, padding=padding)(x)
	res = layers.Activation('relu')(res)
	res = layers.BatchNormalization(momentum=0.8)(res)
	res = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
						strides=strides, padding=padding)(x)
	res = layers.BatchNormalization(momentum=0.8)(res)

	res = layers.Add()([res, x])
	return res

def BuildGenerator():

	ResBlocks = 16
	momentum = 0.8
	input_shape = (224, 224, 3)

	input_layer = keras.Input(shape=input_shape)
	gen1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)
	gen2 = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(gen1)
	gen3 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(gen2)

	res = ResidualBlock(gen3)
	for i in range(ResBlocks-1):
		res = ResidualBlock(res)

	gen4 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(res)
	gen4 = layers.BatchNormalization(momentum=momentum)(gen4)

	gen5 = layers.Add()([gen4, gen3])

	gen6 = layers.UpSampling2D(size=2)(gen5)
	gen6 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen6)
	gen6 = layers.Activation('relu')(gen6)

	gen7 = layers.UpSampling2D(size=2)(gen6)
	gen7 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(gen7)
	gen7 = layers.Activation('relu')(gen7)

	gen8 = layers.Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(gen7)
	output = layers.Activation('tanh')(gen8)

	model = keras.Model(inputs=[input_layer], outputs=[output], name='generator')
	return model