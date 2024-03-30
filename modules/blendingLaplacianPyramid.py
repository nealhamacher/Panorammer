import cv2 as cv
import numpy as np
 
def blendLaplacianPyramid(A, B):
    
    # generate Gaussian pyramid for A
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpA.append(G)
    
    # generate Gaussian pyramid for B
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv.pyrDown(G)
        gpB.append(G)
    
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in range(5,0,-1):
        print(gpA[i-1])
        GE = cv.pyrUp(gpA[i])
        print(GE)
        print(i)
        L = cv.subtract(gpA[i-1],GE)
        lpA.append(L)
    
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in range(5,0,-1):
        GE = cv.pyrUp(gpB[i])
        L = cv.subtract(gpB[i-1],GE)
        lpB.append(L)
    
    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        LS.append(ls)
    
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,6):
        ls_ = cv.pyrUp(ls_)
        ls_ = cv.add(ls_, LS[i])
    
    # image with direct connecting each half
    real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
    
    return ls_