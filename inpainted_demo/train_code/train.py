import matplotlib.pyplot as plt 
import IPython.display as display

from tensorflow.keras.optimizers import Adam

from network import *
from config import *
from ImageController import *

if __name == '__main__':
	
	#Build

	common_optimizer = Adam(0.0002, 0.5)

	#Build VGG19

	vgg = BuildVgg19()
	vgg.trainable = False
	vgg.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

	#Build Discriminator
	discriminator = BuildDiscriminator()
	#discriminator.load_weights("/content/gdrive/My Drive/discriminator_224_withlocal.h5")
	discriminator.compile(loss='mse', optimizer=common_optimizer, metrics=['accuracy'])

	#Build Generator
	generator = BuildGenerator()
	#generator.load_weights("/content/gdrive/My Drive/generator_224_withlocal.h5")

	#Build Adversarial_Model
	input_unpainted_images = keras.Input((224, 224, 3))
	generated_inpainted_images = generator(input_unpainted_images)

	features = vgg(generated_inpainted_images)

	input_local_inpainted_images = layers.experimental.preprocessing.RandomCrop(112, 112)(generated_inpainted_images)

	discriminator.trainable = False
	probs = discriminator((input_local_inpainted_images, generated_inpainted_images))

	adversarial_model = keras.Model(inputs=[input_unpainted_images], 
	                outputs=[probs, features])

	adversarial_model.compile(loss=['binary_crossentropy', 'mse'], 
	              loss_weights=[1e-3, 1], optimizer=common_optimizer)
	#Train

	for epoch in range(epochs):
		for t in range(TD+TC):
			unpainted_images, inpainted_images = sample_images(data_dir=data_dir, batch_size=batch_size, shape_size=shape_size, PointGen=PointGenerator, ns=noise_size)
			unpainted_images = unpainted_images / 127.5 - 1
			inpainted_images = inpainted_images / 127.5 - 1

			generated_inpainted_images = generator.predict(unpainted_images)

			real_labels = np.ones((batch_size, 14, 14, 1))
			fake_labels = np.zeros((batch_size, 14, 14, 1))

			real_local_inapinted = RandCropping(inpainted_images, 112, 112)
			fake_local_inpainted = RandCropping(generated_inpainted_images, 112, 112)

		if t < TD:
			#train discriminator
			d_loss_real = discriminator.train_on_batch([real_local_inapinted, inpainted_images], [real_labels, real_labels])
			d_loss_fake = discriminator.train_on_batch([real_local_inapinted, generated_inpainted_images], [real_labels, fake_labels])
			d_loss_fake += discriminator.train_on_batch([fake_local_inpainted, inpainted_images], [fake_labels, real_labels])
			d_loss_fake += discriminator.train_on_batch([fake_local_inpainted, generated_inpainted_images], [fake_labels, fake_labels])

			#d_loss = 0.5 * np.add(d_loss_real, d_loss_fake / 4)

		else :
			unpainted_images, inpainted_images = sample_images(data_dir=data_dir, batch_size=batch_size, shape_size=shape_size, PointGen=PointGenerator, ns=noise_size)
			unpainted_images = unpainted_images / 127.5 - 1

			image_features = vgg.predict(inpainted_images)
			g_loss = adversarial_model.train_on_batch([unpainted_images], [[real_labels, real_labels], image_features])

		if epoch % 10 == 0:
			unpainted_images, inpainted_images = sample_images(data_dir=data_dir, batch_size=batch_size, shape_size=shape_size, PointGen=PointGenerator, ns=noise_size)
			unpainted_images = unpainted_images / 127.5 - 1
			inpainted_images = inpainted_images / 127.5 - 1

			generated_inpainted_images = generator.predict_on_batch(unpainted_images)

			plt.subplot(1, 3, 1)
			plt.axis('off')
			plt.title('unpainted_image')
			plt.imshow(unpainted_images[0] * 0.5 + 0.5)

			plt.subplot(1, 3, 2)
			plt.axis('off')
			plt.title('inpainted_image')
			plt.imshow(generated_inpainted_images[0] * 0.5 + 0.5)

			plt.subplot(1, 3, 3)
			plt.axis('off')
			plt.title('origin_image')
			plt.imshow(inpainted_images[0] * 0.5 + 0.5)

			display.clear_output(wait=True)
			print("epochs: {}".format(epoch))
			plt.savefig('result')
			plt.show()
			image_cut("result.png", "result.png")

			print('.', end="")

		if epoch % 100 == 0:
			generator.save_weights("/content/gdrive/My Drive/generator_224_withlocal.h5")
			discriminator.save_weights("/content/gdrive/My Drive/discriminator_224_withlocal.h5")