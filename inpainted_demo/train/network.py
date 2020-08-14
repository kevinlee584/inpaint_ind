from tensorflow import keras
from tensorflow.keras import layers

def BuildVgg19():

	vgg = keras.applications.VGG19(weights='imagenet')
	outputs = [vgg.layers[i * 3].output for i in range(1, 4)]
	model = keras.Model([vgg.input], outputs)

	return model

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
	global_input_shape = (224, 224, 3)

	global_input_layer = keras.Input(shape=global_input_shape)
	global_dis1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
	global_dis1 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis1)

	global_dis2 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(global_dis1)
	global_dis2 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis2)
	global_dis2 = layers.BatchNormalization(momentum=momentum)(global_dis2)

	global_dis3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(global_dis2)
	global_dis3 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis3)
	global_dis3 = layers.BatchNormalization(momentum=momentum)(global_dis3)

	global_dis4 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(global_dis3)
	global_dis4 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis4)
	global_dis4 = layers.BatchNormalization(momentum=momentum)(global_dis4)

	global_dis5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(global_dis4)
	global_dis5 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis5)
	global_dis5 = layers.BatchNormalization(momentum=momentum)(global_dis5)

	global_dis6 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(global_dis5)
	global_dis6 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis6)
	global_dis6 = layers.BatchNormalization(momentum=momentum)(global_dis6)

	global_dis7 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(global_dis6)
	global_dis7 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis7)
	global_dis7 = layers.BatchNormalization(momentum=momentum)(global_dis7)

	global_dis8 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(global_dis7)
	global_dis8 = layers.LeakyReLU(alpha=leakyrelu_alpha)(global_dis8)
	global_dis8 = layers.BatchNormalization(momentum=momentum)(global_dis8)

	global_dis9 = layers.Dense(units=1024)(global_dis8)
	global_dis9 = layers.LeakyReLU(alpha=0.2)(global_dis9)

	global_output = layers.Dense(units=1, activation='sigmoid')(global_dis9)


	local_input_shape = (56, 56, 3)

	local_input_layer = keras.Input(shape=local_input_shape)
	local_dis1 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(input_layer)
	local_dis1 = layers.LeakyReLU(alpha=leakyrelu_alpha)(local_dis1)

	local_dis2 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(local_dis1)
	local_dis2 = layers.LeakyReLU(alpha=leakyrelu_alpha)(local_dis2)
	local_dis2 = layers.BatchNormalization(momentum=momentum)(local_dis2)

	local_dis3 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(local_dis2)
	local_dis3 = layers.LeakyReLU(alpha=leakyrelu_alpha)(local_dis3)
	local_dis3 = layers.BatchNormalization(momentum=momentum)(local_dis3)

	local_dis4 = layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(local_dis3)
	local_dis4 = layers.LeakyReLU(alpha=leakyrelu_alpha)(local_dis4)
	local_dis4 = layers.BatchNormalization(momentum=momentum)(local_dis4)

	local_dis5 = layers.Dense(units=1024)(local_dis4)
	local_dis5 = layers.LeakyReLU(alpha=0.2)(local_dis5)

	local_output = layers.Dense(units=1, activation='sigmoid')(local_dis9)

	model = keras.Model(inputs=[local_input_layer, global_input_layer], outputs=[local_output, global_output], name='discriminator')

	return model



