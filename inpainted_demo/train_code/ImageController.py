import cv2 as cv
import glob
import numpy as np


def sample_images(data_dir, batch_size, shape_size, PointGen, ns=0):
  all_images = glob.glob(data_dir)
  image_batch = np.random.choice(all_images, size=batch_size)

  unpainted_images = []
  inpainted_images = []

  for img in image_batch:
    inpainted_img = cv.imread(img)
    # inpainted_img = cv.resize(inpainted_img, (224, 224), interpolation=cv.INTER_CUBIC)
    inpainted_img = inpainted_img[...,::-1]
    inpainted_img = np.rot90(inpainted_img, random.randint(0, 3))
    unpainted_img = inpainted_img.copy()
    shape = unpainted_img.shape

    GenList = [PointGen[random.randint(0, len(PointGen)-1)] for i in range(0, 3)]

    for Gen in GenList:
      for (x, y) in Gen(shape[1], shape[0]):
        noise_size = (int)(ns * (min(shape[0], shape[1]) / 224))
        cv.rectangle(unpainted_img, (x, y), 
               (x+noise_size, y+noise_size), (0, 255, 0), -1)

    inpainted_img = cv.resize(inpainted_img, (224, 224), interpolation=cv.INTER_CUBIC)
    unpainted_img = cv.resize(unpainted_img, (224, 224), interpolation=cv.INTER_CUBIC)

    unpainted_images.append(unpainted_img)
    inpainted_images.append(inpainted_img)
  
  return np.array(unpainted_images), np.array(inpainted_images)

def image_cut(image_dir, name):
  cut_xh = (50, 393)
  cut_yh = (75, 195)
  im = cv.imread(image_dir)
  im = im[cut_yh[0]:cut_yh[1],cut_xh[0]:cut_xh[1]]
  cv.imwrite(name, im)

def tensor_to_image(tensor):
	tensor = tensor*255
	tensor = np.array(tensor, dtype=np.uint8)
	if np.ndim(tensor)>3:
		assert tensor.shape[0] == 1
		tensor = tensor[0]
	return PIL.Image.fromarray(tensor)