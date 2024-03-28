import cv2
import numpy as np
from modules.matching import detectAndDescribe, matchKeypoints

###
# Purpose: Stiches an image onto the final panorama image (colour images)
# Inputs: result - The final panorama image
#         image - The image to stitch
#         matchType - feature matching type, 0 = brute force 1 = k-nearest neighbours
# Returns: The result with the image stitched onto it
###
def stitchColour(result, image, matchType):
    # kpA, fA = detectAndDescribe(np.uint8(result))
    # kpB, fB = detectAndDescribe(image)
    kpA, fA = detectAndDescribe(cv2.cvtColor(np.uint8(result), cv2.COLOR_RGB2GRAY))
    kpB, fB = detectAndDescribe(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    # good_matches, H, temp1, temp2 = matchKeypoints(kpA, kpB, fA, fB, match_type)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, matchType)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                if (imageWarped[i][j][k] != 0 and result[i][j][k] == 0):
                    if not has_intersected:
                        intersect_points.append((j, i))
                        has_intersected = True
                    result[i][j][k] = imageWarped[i][j][k]
    return result


###
# Purpose: Stiches an image onto the final panorama image (grayscale images)
# Inputs: result - The final panorama image
#         image - The image to stitch
#         matchType - feature matching type, 0 = brute force 1 = k-nearest neighbours
# Returns: The result with the image stitched onto it
###
def stitchGrey(result, image, match_type):
    kpA, fA = detectAndDescribe(np.uint8(result))
    kpB, fB = detectAndDescribe(image)
    # good_matches, H, temp1, temp2 = matchKeypoints(kpA, kpB, fA, fB, match_type)
    _, H, _, _ = matchKeypoints(kpA, kpB, fA, fB, match_type)

    imageWarped = cv2.warpPerspective(image, H, (result.shape[1], result.shape[0]))
    intersect_points = []

    for i in range(result.shape[0]):
        has_intersected = False
        for j in range(result.shape[1]):
            if (imageWarped[i][j] != 0 and result[i][j] == 0):
                if not has_intersected:
                    intersect_points.append((j, i))
                    has_intersected = True
                result[i][j] = imageWarped[i][j]
    return result