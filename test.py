from modules.blendingPoisson import blendingPoisson
from modules.blendingLaplacianPyramid import blendLaplacianPyramid
import cv2

im1 = cv2.cvtColor(cv2.imread('images/macew1.jpg'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('images/macew3.jpg'), cv2.COLOR_BGR2RGB)


result = blendLaplacianPyramid(im1, im2)

cv2.imshow("BLENDED", result)
cv2.waitKey()