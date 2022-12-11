import numpy as np 
import cv2
import open3d  as o3d
import json
import glob

from matplotlib import pyplot as plt
config = json.loads(open('config.json', 'r').read())



pathL =glob.glob(config['examples_directory'] + "Esquerda/*.jpg")
pathR = glob.glob(config['examples_directory'] + "Direita/*.jpg")

print("Reading parameters ......")
cv_file = cv2.FileStorage(config['calibration_files_directory']+"/rectification_map.yml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
Q = cv_file.getNode("Q").mat()
cv_file.release()

config = json.loads(open('stereo_params_2.json', 'r').read())

def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(points_3D[x,y,2],disparity[x,y])

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(disparity, 
                    str(points_3D[x,y,2]), (x,y), font,
                    0.7, (0, 0, 255), 2)
        cv2.imshow('dispv', disparity)
 

 

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = config['blockSize']  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=config['minDisparity']-10,
        numDisparities=config['numDisparities'],  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=12,
        uniquenessRatio=config['uniquenessRatio'],
        speckleWindowSize=config['speckleWindowSize'],
        speckleRange=config['speckleRange'],
        preFilterCap=config['preFilterCap'],
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = config['lmbda']
    sigma = config['sigma']/10


    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)#.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)#.astype(np.float32)/16
    # disparity=displ.astype(np.float32)
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    
    # cv2.imshow("R",disparity)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX);
    # norm = np.zeros((800,800))
    # filteredImg = cv2.normalize(filteredImg,  norm, 255, 0, cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg 


for imgLeft, imgRight in zip(pathL, pathR):
        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
    
    
        # cv2.waitKey(0)
            
        
        Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
        Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        
        Left_nice_2= Left_nice
        Right_nice_2= Right_nice
        
       
        Right_nice = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        Left_nice = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        cv2.imshow("L_1",Left_nice_2)
        cv2.imshow("R_1",Right_nice_2)
        cv2.imshow("L",imgL)
        cv2.imshow("R",imgR)
        
        smaller_size=(int(Right_nice.shape[0]/2),int(Right_nice.shape[1]/2))
        # left_image_rect_downscaled = cv2.resize(Left_nice, smaller_size, interpolation=cv2.INTER_AREA)
        
        # right_image_rect_downscaled = cv2.resize(Right_nice, smaller_size, interpolation=cv2.INTER_AREA)
        
        disparity = depth_map(Left_nice, Right_nice)
        # disparity = cv2.medianBlur(disparity,21)
        cv2.imshow("dispv",disparity)

        output = Right_nice_2.copy()
        output[:,:,0] = Right_nice_2[:,:,0]
        output[:,:,1] = Right_nice_2[:,:,1]
        output[:,:,2] = Left_nice_2[:,:,2]
        # output = Left_nice+Right_nice
        # output = cv2.resize(output,(700,700))
        
        cv2.imshow("3D movie",output)
        points_3D = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=False)
        # print(points_3D[1000,1400].shape)
        # print(points_3D[1000,1401][2])
        cv2.setMouseCallback('dispv', click_event)
        cv2.waitKey(0)
        break

            
    