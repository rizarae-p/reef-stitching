import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

def detectAndDescribe(image, method=None):
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

GOOD_MATCH_PERCENT = 0.6
bf = createMatcher("orb",True)
d = str(input("img directory: "))
files = sorted(glob.glob(d+"*.jpg"))
trainImg_fname = files[0]
homography = []
width,height = 0,0
for i in files[1:]:
    # if trainImg_fname != i:
    #     trainImg = cv2.imread("out.jpg")
    # else:
    trainImg = cv2.imread(trainImg_fname)
    queryImg = cv2.imread(i)
    # trainImg_fname = i
    print(i)

    (kpsA, featuresA) = detectAndDescribe(trainImg,"orb")
    (kpsB, featuresB) = detectAndDescribe(queryImg,"orb")
    matches = bf.match(featuresA,featuresB)
    matches = sorted(matches, key = lambda x:x.distance)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    width = trainImg.shape[1] + queryImg.shape[1]//2
    height = trainImg.shape[0] + queryImg.shape[0]//2
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    matches = np.array(matches)
    for i, match in enumerate(matches):
        points1[i, :] = kpsA[match.queryIdx].pt
        points2[i, :] = kpsB[match.trainIdx].pt
       
      # Find homography
    H, mask = cv2.findHomography(points1, points2,cv2.RANSAC)
    homography.append(H)
    # result = cv2.warpPerspective(trainImg, H, (width, height))
    # result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    # indices_r = np.all(result == [0,0,0], axis=1)[:,0]
    # indices_c = np.all(result == [0,0,0], axis=0)[:,0]
    # r = np.where(indices_r == True)[0]
    # c = np.where(indices_c == True)[0]
    # min_r = min(r) if len(r) > 0 else trainImg.shape[0]
    # min_c = min(c) if len(c) > 0 else trainImg.shape[1]
    # new_res = result[:min_r,:min_c]
    # cv2.imwrite("out.jpg",new_res)
    # cv2.imwrite("res/out"+str(ctr)+".jpg",new_res)
    # print(new_res.shape)


# def blend(alpha,src1,src2):
#     beta = (1.0 - alpha)
#     dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
#     return dst

# loop for iterating over homographies
prev = []
for i in range(len(homography)):
    img_mat = cv2.imread(files[i+1],0)
    H = homography[i]
    result = cv2.warpPerspective(img_mat, H, (width, height))
    if len(prev) != 0:
        print (prev.shape == result.shape)
        # result = blend(0.1,prev,result)
        # bool_mat = prev == 0
        bool_mat = np.all(prev == 0)
        # print(bool_mat.reshape((img_mat.shape)))
        result = prev[bool_mat] + result[bool_mat]
    prev = result
    cv2.imwrite("out.jpg",result)
    cv2.imwrite("res/out"+str(i)+".jpg",result)





#extra code
    # result[0:img_mat.shape[0], 0:img_mat.shape[1]] = img_mat
    # indices_r = np.all(result == [0,0,0], axis=1)[:,0]
    # indices_c = np.all(result == [0,0,0], axis=0)[:,0]
    # r = np.where(indices_r == True)[0]
    # c = np.where(indices_c == True)[0]
    # min_r = min(r) if len(r) > 0 else img_mat.shape[0]
    # min_c = min(c) if len(c) > 0 else img_mat.shape[1]
    # new_res = result[:min_r,:min_c]