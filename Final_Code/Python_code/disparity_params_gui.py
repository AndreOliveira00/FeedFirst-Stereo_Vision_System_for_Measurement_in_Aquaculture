import numpy as np 
import cv2
import open3d  as o3d
import json
import glob
from matplotlib import pyplot as plt

config = json.loads(open('config.json', 'r').read())

CamL_id = "stereoL.mp4"
CamR_id= "stereoR.mp4"

examples_directory = config['examples_directory']
calibration_files_directory = config['calibration_files_directory']

pathL =glob.glob(config['examples_directory'] + "Esquerda/*.jpg")
pathR = glob.glob(config['examples_directory'] + "Direita/*.jpg")


print("Reading parameters ......")
cv_file = cv2.FileStorage(calibration_files_directory+"/rectification_map.yml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

config = json.loads(open('stereo_params_2.json', 'r').read())
print(config)


numDisparities = config['numDisparities']
blockSize = config['blockSize']
preFilterType = config['preFilterType']
preFilterSize = config['preFilterSize']
preFilterCap = config['preFilterCap']
textureThreshold = config['textureThreshold']
uniquenessRatio = config['uniquenessRatio']
speckleRange = config['speckleRange']
speckleWindowSize = config['speckleWindowSize']
disp12MaxDiff = config['disp12MaxDiff']
minDisparity = config['minDisparity']
lmbda = config['lmbda']
sigma = config['sigma']
# visual_multiplier = config['visual_multiplier']


cv2.namedWindow('SGBM_params',cv2.WINDOW_NORMAL)
cv2.resizeWindow('SGBM_params',1000,200)

def nothing(x):
    pass

 
def depth_map(imgL, imgR):
    # Depth map calculation. 
    # Works with SGBM and WLS. 
    # Need rectified images, returns depth map ( left to right disparity ) 
    # SGBM Parameters 
    window_size = cv2.getTrackbarPos('blockSize','SGBM_params')  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=cv2.getTrackbarPos('minDisparity','SGBM_params')-10,
        numDisparities=cv2.getTrackbarPos('numDisparities','SGBM_params')*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=cv2.getTrackbarPos('disp12MaxDiff','SGBM_params'),
        uniquenessRatio=cv2.getTrackbarPos('uniquenessRatio','SGBM_params'),
        speckleWindowSize=cv2.getTrackbarPos('speckleWindowSize','SGBM_params'),
        speckleRange=cv2.getTrackbarPos('speckleRange','SGBM_params'),
        preFilterCap=cv2.getTrackbarPos('preFilterCap','SGBM_params'),
        mode=cv2.STEREO_SGBM_MODE_HH
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lamda = cv2.getTrackbarPos('lamda','SGBM_params')
    sigma = cv2.getTrackbarPos('sigma','SGBM_params')/10
    print(lamda)
    print(sigma)
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lamda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)#.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)#.astype(np.float32)/16
    
    # disparity=displ.astype(np.float32)/16-10
    # disparity = (disparity)/cv2.getTrackbarPos('numDisparities','SGBM_params')*16
    # disparity = (displ/16.0 +1)/cv2.getTrackbarPos('numDisparities','SGBM_params')*16
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    
    cv2.imshow("Disparit1y",displ)
    cv2.imshow("Disparit2y",dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=255, alpha=0, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

cv2.createTrackbar('numDisparities','SGBM_params',int(numDisparities/16),25,nothing)
cv2.createTrackbar('blockSize','SGBM_params',int(blockSize),50,nothing)
cv2.createTrackbar('preFilterType','SGBM_params',preFilterType,1,nothing)
cv2.createTrackbar('preFilterSize','SGBM_params',int((preFilterSize-5)/2),25,nothing)
cv2.createTrackbar('preFilterCap','SGBM_params',preFilterCap,400,nothing)
cv2.createTrackbar('textureThreshold','SGBM_params',textureThreshold,100,nothing)
cv2.createTrackbar('uniquenessRatio','SGBM_params',uniquenessRatio,200,nothing)
cv2.createTrackbar('speckleRange','SGBM_params',speckleRange,400,nothing)
cv2.createTrackbar('speckleWindowSize','SGBM_params',int(speckleWindowSize/2),25,nothing)
cv2.createTrackbar('disp12MaxDiff','SGBM_params',disp12MaxDiff,40,nothing)
cv2.createTrackbar('minDisparity','SGBM_params',minDisparity,25,nothing)
cv2.createTrackbar('sigma','SGBM_params',sigma,100,nothing)
cv2.createTrackbar('lamda','SGBM_params',lmbda,150000,nothing)
# cv2.createTrackbar('visual_multiplier','SGBM_params',visual_multiplier,25,nothing)
# cv2.createButton("Back",nothing,None,cv2.QT_PUSH_BUTTON,1)
stereo = cv2.StereoBM_create()
o=0

for imgLeft, imgRight in zip(pathL, pathR):
        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
       
        o=o+1
    
        cv2.waitKey(0)
        # disparity with colors
        # plt.close('all')      
        
        Left_nice= cv2.remap(imgL,Left_Stereo_Map_x,Left_Stereo_Map_y, cv2.INTER_LINEAR,cv2.BORDER_CONSTANT)
        Right_nice= cv2.remap(imgR,Right_Stereo_Map_x,Right_Stereo_Map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        
        Left_nice_rect= Left_nice
        Right_nice_rect= Right_nice
        
       
        Right_nice = cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        Left_nice = cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)
        cv2.imshow("L_rect",Left_nice_rect)
        # cv2.imshow("R_1", )
        cv2.imshow("L_unrect",imgL)
        # cv2.imshow("R",imgR)
        #while True:
        
        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities','SGBM_params')*16
        blockSize = cv2.getTrackbarPos('blockSize','SGBM_params')
        preFilterType = cv2.getTrackbarPos('preFilterType','SGBM_params')
        preFilterSize = cv2.getTrackbarPos('preFilterSize','SGBM_params')
        reFilterCap = cv2.getTrackbarPos('preFilterCap','SGBM_params')
        textureThreshold = cv2.getTrackbarPos('textureThreshold','SGBM_params')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','SGBM_params')
        speckleRange = cv2.getTrackbarPos('speckleRange','SGBM_params')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','SGBM_params')
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','SGBM_params')
        minDisparity = cv2.getTrackbarPos('minDisparity','SGBM_params')
        lamda=cv2.getTrackbarPos('lamda','SGBM_params')
        sigma=cv2.getTrackbarPos('sigma','SGBM_params')	
         
        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        # stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)
        
        
        # smaller_size=(int(Right_nice.shape[0]/2),int(Right_nice.shape[1]/2))
        # left_image_rect_downscaled = cv2.resize(Left_nice, smaller_size, interpolation=cv2.INTER_AREA)
             # right_image_rect_downscaled = cv2.resize(Right_nice, smaller_size, interpolation=cv2.INTER_AREA)
        disparity= depth_map(Left_nice, Right_nice)
        # disparity = stereo.compute(Left_nice, Right_nice)
        # disparity = disparity.astype(np.float32)
        # disparity = (disparity/16.0 - minDisparity)/numDisparities
        # disparity_S = cv2.resize(disparity, (int(disparity.shape[0]/5*3), int(disparity.shape[1]/5*3)))
        cv2.imshow("dispv",disparity)
        
        output = Right_nice_rect.copy()
        output[:,:,0] = Right_nice_rect[:,:,0]
        output[:,:,1] = Right_nice_rect[:,:,1]
        output[:,:,2] = Left_nice_rect[:,:,2]
        # output = Left_nice+Right_nice
        # output = cv2.resize(output,(700,700))
             # cv2.namedWindow("3D movie",cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("3D movie",int(disparity.shape[0]/3),int(disparity.shape[1]/3))
        cv2.imshow("3D movie",output)
        cv2.waitKey(1)
        
        stereo_params ={
            "numDisparities" : numDisparities,
            "blockSize" : blockSize,
            "preFilterType" : preFilterType,
            "preFilterSize" : preFilterSize,
            'preFilterCap' : preFilterCap,
            'textureThreshold':textureThreshold,
            'uniquenessRatio':uniquenessRatio,
            'speckleRange':speckleRange,
            'speckleWindowSize':speckleWindowSize,
            'disp12MaxDiff':disp12MaxDiff,
            'minDisparity':minDisparity,
            'lmbda':lmbda,
            'sigma':sigma         
        }   
        with open('stereo_params.json', 'w') as f:
            json.dump(stereo_params, f)
            
            cv2.waitKey(10)