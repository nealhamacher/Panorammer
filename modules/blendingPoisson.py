# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2


def blendingPoisson(source, destination):
  # create an all "White" mask: 255, if black mask is 0
  mask = np.zeros(source.shape, source.dtype) 
  # navigate the source img location
  if source.ndim == 3:
    width, height, _ = destination.shape
    # for i in range(0, source.shape[0]):
    #   for j in range(0, source.shape[1]):
    #     if np.any(source[i][j][:]):
    #       mask[i][j][:] = 255
  else:
    width, height = destination.shape
    # for i in range(0, source.shape[0]):
    #   for j in range(0, source.shape[1]):
    #     if source[i][j] != 0:
    #       mask[i][j] = 255

  mask[:,0:source.shape[1]//2,:] = 255
  print(source.ndim)
  print(destination.ndim)
  cv2.imshow("Mask", mask)
  cv2.waitKey()
  center = (height//2, width//2)
  # using built-in funtion `cv2.seamlessClone` to acommpulish Poisson Image
  blended = cv2.seamlessClone(cv2.cvtColor(source, cv2.COLOR_RGB2BGR), cv2.cvtColor(destination, cv2.COLOR_RGB2BGR), mask, center, 1) # cv::MIXED_CLONE = 2
  return blended