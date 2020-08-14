import cv2 as cv
import random

class noise_generator:
  def __init__(self, n):
    self.n = n

  def __call__(self,width, high):
    return self.gen(width, high)

  def gen(self, width, high):
    pass

class CycleNoise(noise_generator):
  def __init__(self, n, MinLen, MaxLen):
    super().__init__(n)
    self.MinLen = MinLen
    self.MaxLen = MaxLen

  def gen(self, width, high):
    fn = (((random.randint(0, width), random.randint(0, high))) for i in range(self.n))

    start_x = random.randint(0, width)
    start_y = random.randint(0, high)
    R = random.randint(self.MinLen, self.MaxLen) * (min(width, high) / 224)

    return [(x, y) for x, y in fn if ((x-start_x)**2 + (y-start_y)**2 <= R**2)]

class RectangleNoise(noise_generator):
  def __init__(self, n, MinLen, MaxLen):
    super().__init__(n)
    self.MinLen = MinLen
    self.MaxLen = MaxLen

  def gen(self, width, high):
    fn = (((random.randint(0, width), random.randint(0, high))) for i in range(self.n))
    ML = (int)(self.MinLen * (min(width, high) / 224))

    start_x = random.randint(0, width)
    start_y = random.randint(0, high)
    end_x = start_x + random.randint(self.MinLen, self.MaxLen) * (min(width, high) / 224)
    end_y = start_y + random.randint(self.MinLen, self.MaxLen) * (min(width, high) / 224)

    return [(x, y) for x, y in fn if (start_x < x < end_x and start_y < y < end_y)]
