import cv2
import numpy as np
from modules.matching import detectAndDescribe, matchKeypoints

###
# Purpose: Stiches an image onto the final panorama image (grayscale images). 
#          Does not blend the images (takes pixel value from result where the 
#          images) overlap
# Inputs: result - The final panorama image
#         image - The image to stitch
#         matchType - feature matching type, 0 = brute force 1 = k-nearest neighbours
# Returns: The result with the image stitched onto it
###
def __stitchGray(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    print(imageWarped.shape)
    print(result.shape)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (imageWarped[i][j] != 0 and result[i][j] == 0):
                result[i][j] = imageWarped[i][j]
    
    return result
                

###
# Purpose: Stiches an image onto the final panorama image (colour images). Does
#          not blend the images (takes pixel value from result where the images)
#          overlap
# Inputs: result - The final panorama image
#         image - The image to stitch
#         match_type - feature matching type, 0 = brute force 1 = k-nearest neighbours
#         Returns: The result with the image stitched onto it
###
def __stitchColour(result, image, matchType):
    kpA, fA = detectAndDescribe(cv2.cvtColor(np.uint8(result), cv2.COLOR_RGB2GRAY))
    kpB, fB = detectAndDescribe(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, matchType)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageWarped[i][j][k] != 0 and result[i][j][k] == 0):
                    result[i][j][k] = imageWarped[i][j][k]     

    return result


def __stitchGrayAvg(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if (imageWarped[i][j] != 0):
                if (result[i][j] != 0):
                    result[i][j] = (imageWarped[i][j] + result[i][j]) // 2
                else:
                    result[i][j] = imageWarped[i][j]
            # Otherwise imageWarped[i][j] == 0 and result doesn't change

    return result


def __stitchColourAvg(result, image, matchType):
    kpA, fA = detectAndDescribe(cv2.cvtColor(np.uint8(result), cv2.COLOR_RGB2GRAY))
    kpB, fB = detectAndDescribe(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, matchType)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageWarped[i][j][k] != 0):
                    if (result[i][j][k] != 0):
                        result[i][j][k] = (imageWarped[i][j][k] + result[i][j][k]) // 2
                    else:
                        result[i][j][k] = imageWarped[i][j][k]  

    return result


def stitchGrayWeighted(result, image, matchType):
    pass


###
# Purpose: Main function from this module. Stitches two images together. Calls 
#          one of the other functions to perform the stitchig, depending on the 
#          colour type of the images and the blending method used
# Inputs: TODO 
# Returns: The stitched together result.
###
def stitch(result, image, colour_type, match_type=0, blend_type=0):
    if colour_type == 'rgb':
        if blend_type == 0:
            result = __stitchColour(result, image, match_type)
        elif blend_type == 1:
            result = __stitchColourAvg(result, image, match_type)
    else: 
        if blend_type == 0:
            result = __stitchGray(result, image, match_type)
        elif blend_type == 1:
            result = __stitchGrayAvg(result, image, match_type)

    return result