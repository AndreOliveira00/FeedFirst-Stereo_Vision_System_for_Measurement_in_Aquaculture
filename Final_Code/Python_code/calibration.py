import cv2
import numpy as np
import json
import glob
import os
from tqdm import tqdm
import scipy.io
mat = scipy.io.loadmat('stereoParams.mat')

config = json.loads(open('config.json', 'r').read())

pattern_size = (config['pattern_rows'], config['pattern_columns'])
square_size_mm = config['square_size_mm']
calibration_files_directory = config['calibration_files_directory']
pathL =glob.glob(config['calibration_files_directory'] + "stereoL/*.jpg")
pathR = glob.glob(config['calibration_files_directory'] + "stereoR/*.jpg")
# samples_num=len(os.listdir(pathL))
# print(pathL)
# imagesLeft = glob.glob('images/stereoLeft/*.bmp')
# imagesRight = glob.glob('images/stereoRight/*.png')

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, square_size_mm, 0.001)


# objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
# objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)

# img_ptsL = []
# img_ptsR = []
# obj_pts = []

# for imgLeft, imgRight in zip(pathL, pathR):
#      imgL = cv2.imread(imgLeft)
#      imgR = cv2.imread(imgRight)
#      # print(pathL)
#      # cv2.imshow('l',imgL)
#      # cv2.waitKey(0)
#      imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#      imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
     
#      outputL = imgL.copy()
#      outputR = imgR.copy()
     
#      retR, cornersR =  cv2.findChessboardCorners(outputR,pattern_size,None)
#      retL, cornersL = cv2.findChessboardCorners(outputL,pattern_size,None)
#      print(retR,imgLeft)
     
#      if retR and retL:
#          obj_pts.append(objp)
#          img_ptsL.append(cornersL)
#          img_ptsR.append(cornersR)
#          # ----------- DRAW ---------------
#          cv2.cornerSubPix(imgR_gray,cornersR,(11,11),(-1,-1),criteria)
#          cv2.cornerSubPix(imgL_gray,cornersL,(11,11),(-1,-1),criteria)
#          cv2.drawChessboardCorners(outputR,pattern_size,cornersR,retR)
#          cv2.drawChessboardCorners(outputL,pattern_size,cornersL,retL)
#          cv2.imshow('cornersR',outputR)
#          cv2.imshow('cornersL',outputL)
         
#          cv2.waitKey(10)
         
# # Calibrating left camera
# retL, camMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts,img_ptsL,imgL_gray.shape[::-1],None,None)
# hL,wL= imgL_gray.shape[:2]
# new_camMatrixL, roiL= cv2.getOptimalNewCameraMatrix(camMatrixL,distL,(wL,hL),1,(wL,hL))
# # new_camMatrixL=camMatrixL

# # Calibrating right camera
# retR, camMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts,img_ptsR,imgR_gray.shape[::-1],None,None)
# hR,wR= imgR_gray.shape[:2]
# new_camMatrixR, roiR= cv2.getOptimalNewCameraMatrix(camMatrixR,distR,(wR,hR),1,(wR,hR))
# # new_camMatrixR=camMatrixR
# print("Camera Right Calibrated: ",  retR) 
# print("\nRight Camera Matrix:\n", camMatrixR) 
# print("\nRight Distortion Parameters:\n", distR) 
# print("\nRight Rotation Vectors:\n", rvecsR) 
# print("\nRight Translation Vectors:\n", tvecsR) 

# #----------------- STEREO CALIBRATION -----------------#

# flags = 0
# flags |= cv2.CALIB_FIX_INTRINSIC

# criteria_stereo = criteria

# #calculate Essential and Fundamenatl matrix
# retS, new_camMatrixL, distL, new_camMatrixR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts,
#                                                                                     img_ptsL,
#                                                                                     img_ptsR,
#                                                                                     new_camMatrixL,
#                                                                                     distL,
#                                                                                     new_camMatrixR,
#                                                                                     distR,
#                                                                                     imgL_gray.shape[::-1],
#                                                                                     criteria_stereo,
#                                                                                     flags)

mat = scipy.io.loadmat('stereoParams.mat')
camMatrixL=mat["camMatrixL"]
distL=mat["distL"]
camMatrixR=mat["camMatrixR"]
distR=mat["distR"]
Trns=mat["Trns"]
Rot=mat["Rot"]
img_size=mat["img_size"]

# StereoRectify function
rectify_scale= 1 # if 0 image croped, if 1 image not croped
rect_L, rect_R, proj_mat_L, proj_mat_R, Q, roiL, roiR= cv2.stereoRectify(camMatrixL, distL, camMatrixR, distR,
                                                                         img_size[0], Rot, Trns, None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, 0)


# Compute the rectification map 
Left_Stereo_Map= cv2.initUndistortRectifyMap(camMatrixL, distL, rect_L, proj_mat_L,
                                             img_size[0], cv2.CV_32FC1)
Right_Stereo_Map= cv2.initUndistortRectifyMap(camMatrixR, distR, rect_R, proj_mat_R,
                                              img_size[0], cv2.CV_32FC1)

cv_file = cv2.FileStorage(calibration_files_directory+"/rectification_map.yml", cv2.FILE_STORAGE_WRITE)
cv_file.write("Left_Stereo_Map_x",Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y",Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x",Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y",Right_Stereo_Map[1])
cv_file.write('Q', Q)
cv_file.release()

print("Rectification map saved at:",calibration_files_directory)

# Reprojection Error
# mean_error = 0
# for i in range(len(obj_pts)):
#     imgpointsL2, _ = cv2.projectPoints(obj_pts[i], rvecsL[i], tvecsL[i], camMatrixL, distL)
#     error = cv2.norm(img_ptsL[i], imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
#     mean_error += error

# print( "Total Error Left Camera: {}".format(mean_error/len(obj_pts)) )

# mean_error = 0

# for i in range(len(obj_pts)):
#     imgpointsR2, _ = cv2.projectPoints(obj_pts[i], rvecsR[i], tvecsR[i], camMatrixR, distR)
#     error = cv2.norm(img_ptsR[i], imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
#     mean_error += error
# print( "Total Error Right Camera: {}".format(mean_error/len(obj_pts)) )
