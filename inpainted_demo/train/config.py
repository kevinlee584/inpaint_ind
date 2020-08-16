from RandomNoise import *

sdata_dir = '/content/gdrive/My Drive/paint/*.jpg'
epochs = 201
TD = 100
TC = 1
batch_size = 16
noise_num = 800
noise_size = 12
MinLen = 30
MaxLen = 40
shape_size = (224, 224, 3)

PointGenerator = [
  CycleNoise(noise_num, MinLen, MaxLen), 
  RectangleNoise(noise_num, MinLen, MaxLen)]