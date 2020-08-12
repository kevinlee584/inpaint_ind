from model.model import *
import cv2 as cv
import glob
import numpy as np
import os
import PIL.Image


def sample_images(imgs_dir):
	all_images = glob.glob(imgs_dir+"/*.jpg")

	unpainted_images = []
	images_detail = []

	for img_dir in all_images:
		img = cv.imread(img_dir)
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #BGR 轉 RGB
		shape = img.shape
		img = cv.resize(img, (224, 224), interpolation=cv.INTER_CUBIC)
		
		unpainted_images.append(img)
		images_detail.append({
				'Shape' : (shape[1], shape[0]), 
				'Dir': img_dir
			})

	return np.array(unpainted_images), images_detail

def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)

if __name__ == "__main__":

	cwd = os.getcwd()
	unpainted_image_path = cwd + "/Image/Unpainted"
	inpainted_image_path = cwd + "/Image/Inpainted"

	generator = BuildGenerator()
	generator.load_weights(cwd + "/model/ModelWeight/generator_224.h5")

	unpainted_images, images_detail = sample_images(unpainted_image_path)
	unpainted_images = unpainted_images / 127.5 - 1

	generated_inpainted_images = generator.predict_on_batch(unpainted_images)

	for img, detail in zip(generated_inpainted_images, images_detail):
		im = tensor_to_image(img * 0.5 + 0.5).resize(detail['Shape'], PIL.Image.BILINEAR )
		im.save(detail['Dir'].replace(unpainted_image_path, inpainted_image_path))
