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


def BuildDiscriminator():

	leakyrelu_alpha = 0.2
	momentum = 0.8
	input_shape = (224, 224, 3)

	input_layer = keras.Input(shape=input_shape)
	dis1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
	dis1 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis1)

	dis2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
	dis2 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis2)
	dis2 = layers.BatchNormalization(momentum=momentum)(dis2)

	dis3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
	dis3 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis3)
	dis3 = layers.BatchNormalization(momentum=momentum)(dis3)

	dis4 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
	dis4 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis4)
	dis4 = layers.BatchNormalization(momentum=momentum)(dis4)

	dis5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(dis4)
	dis5 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis5)
	dis5 = layers.BatchNormalization(momentum=momentum)(dis5)

	dis6 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
	dis6 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis6)
	dis6 = layers.BatchNormalization(momentum=momentum)(dis6)

	dis7 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
	dis7 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis7)
	dis7 = layers.BatchNormalization(momentum=momentum)(dis7)

	dis8 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
	dis8 = layers.LeakyReLU(alpha=leakyrelu_alpha)(dis8)
	dis8 = layers.BatchNormalization(momentum=momentum)(dis8)

	dis9 = layers.Dense(units=1024)(dis8)
	dis9 = layers.LeakyReLU(alpha=0.2)(dis9)	
	output = layers.Dense(units=1, activation='sigmoid')(dis9)
	model = keras.Model(inputs=[input_layer], outputs=[output], name='discriminator')
	return model