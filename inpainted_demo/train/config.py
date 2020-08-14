from RandomNoise import *

shape_size = (224, 224, 3)

data_dir = '/content/gdrive/My Drive/paint/*.jpg'
epochs = 20001
batch_size = 16
noise_num = 800
noise_size = 12
MinLen = 30
MaxLen = 40

PointGenerator = [
  CycleNoise(noise_num, MinLen, MaxLen), 
  RectangleNoise(noise_num, MinLen, MaxLen)]