import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2
import glob

# labels of joints:
# 1.base of spine 2.middle of spine 3.neck 4.head 5.left shoulder 
# 6.left elbow 7.left wrist 8.left hand 9.right shoulder
# 10.right elbow 11.right wrist 12.right hand 13.left hip
# 14.left knee 15.left ankle 16.left foot 17.right hip 
# 18.right knee 19.right ankle 20.right foot 21.spine 
# 22.tip of left hand 23.left thumb 24.tip of right hand
# 25.right thumb. In arrays all numbers are subtracted by one
SPINE_POINTS = [0, 1, 20, 2, 3]
ARM_POINTS = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
LEG_POINTS = [19, 18, 17, 16, 0, 12, 13, 14, 15]
BODY_POINTS = [SPINE_POINTS, ARM_POINTS, LEG_POINTS]

# the NTU RGB skeleton files content:
# First line - number of recorded frames
# And then a sequence starts:
# 1. number of skeletons in a frame
# 2. one line containing information about the skeleton
# 3. number of joint points
# 4. 25 lines of each point info
# point info contains: x,y,z,depthX, depthY,colorX,colorY,
# orientationW,orientationX,orientationY,orientationZ,trackingState
SKELETON_INFORMATIONS = ['bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState']
                    
POINT_INFORMATIONS = ['x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                     'orientationW', 'orientationX', 'orientationY',
                    'orientationZ', 'trackingState']
                    
        
def readSkeleton(fileName):
    with open(fileName, 'r') as file:
        skeletonFileData = dict()
        skeletonFileData["numberOfFrames"] = int(file.readline())
        skeletonFileData["frameInfo"] = []
        for frame in range(skeletonFileData["numberOfFrames"]):
            frameInfo = dict()
            frameInfo["numberOfSkeletons"] = int(file.readline())
            frameInfo["skeletonInfo"] = []
            for skeleton in range(frameInfo["numberOfSkeletons"]):
                skeletonInfo = dict()
                for key, value in zip(SKELETON_INFORMATIONS, file.readline().split()):
                    skeletonInfo[key] = value
                
                skeletonInfo["numberOfJoints"] = int(file.readline())
                skeletonInfo["jointInfo"] = []
                for jointValues in range(skeletonInfo["numberOfJoints"]):
                    jointInfo = dict()
                    for key,value in zip(POINT_INFORMATIONS, file.readline().split()):
                        jointInfo[key] = value
                    skeletonInfo["jointInfo"].append(jointInfo)
                frameInfo["skeletonInfo"].append(skeletonInfo)
            skeletonFileData["frameInfo"].append(frameInfo)
    return skeletonFileData

# getPoints Function get as input parameter '.skeleton' file data and return XYZ Coordinates or XYRGB or XYDepth depending on the mode
def getPoints(skletonFileData, mode, maxSkeletons=2, numberOfJoints=25):
    joint = dict()
    if mode == 'depth':
        returnPoints = np.zeros((2, skletonFileData["numberOfFrames"], numberOfJoints, maxSkeletons))
        for countFrame, frameInfo in enumerate(skletonFileData["frameInfo"]):
            for countSkeletons, skeletonInfo in enumerate(frameInfo["skeletonInfo"]):
                for countJoint, joint in enumerate(skeletonInfo["jointInfo"]):
                    if (countSkeletons < maxSkeletons and countJoint < numberOfJoints):
                        returnPoints[:, countFrame, countJoint, countSkeletons] = [joint['depthX'], joint['depthY']]
                    else:
                        pass
        return returnPoints
    elif mode == 'rgb':
        returnPoints = np.zeros((2, skletonFileData["numberOfFrames"], numberOfJoints, maxSkeletons))
        for countFrame, frameInfo in enumerate(skletonFileData["frameInfo"]):
            for countSkeletons, skeletonInfo in enumerate(frameInfo["skeletonInfo"]):
                for countJoint, joint in enumerate(skeletonInfo["jointInfo"]):
                    if (countSkeletons < maxSkeletons and countJoint < numberOfJoints):
                        returnPoints[:, countFrame, countJoint, countSkeletons] =  [joint['colorX'], joint['colorY']]
                    else:
                        pass
        return returnPoints
    elif mode == "xyz":
        returnPoints = np.zeros((3, skletonFileData["numberOfFrames"], numberOfJoints, maxSkeletons))
        for countFrame, frameInfo in enumerate(skletonFileData["frameInfo"]):
            for countSkeletons, skeletonInfo in enumerate(frameInfo["skeletonInfo"]):
                for countJoint, joint in enumerate(skeletonInfo["jointInfo"]):
                    if (countSkeletons < maxSkeletons and countJoint < numberOfJoints):
                        returnPoints[:, countFrame, countJoint, countSkeletons] = [joint['x'], joint['y'], joint['z']]
                    else:
                        pass
        return returnPoints

# setCenterPoint 3DPlot normalization function
def setCenterPoint(skletonFileData):
    firstPoint = skletonFileData[0, :, 0, :]
    firstPointx = np.mean(firstPoint[:, 0])
    firstPointy = np.mean(firstPoint[:, 1])
    firstPointz = np.mean(firstPoint[:, 2])

    averageCenter = np.array([firstPointx, firstPointy, firstPointz])
    # reset data to have overlapping points in plot
    skletonFileData = skletonFileData - averageCenter

    return skletonFileData

def show3DPlot(pointsData):
    pltFigure = plt.figure()
    axes = Axes3D(pltFigure)
    elevationAngle = 20
    azimuthAngle = -45
    axes.view_init(elevationAngle, azimuthAngle)
    plt.ion()

    skeletonData = np.transpose(pointsData, (3, 1, 2, 0)) 
    skeletonData = setCenterPoint(skeletonData)

    # show frame by frame 3d skeleton
    for frame in range(skeletonData.shape[1]):
        plt.cla()
        axes.set_xlim3d([-1, 1])
        axes.set_ylim3d([-1, 1])
        axes.set_zlim3d([-0.8, 0.8])

        x = skeletonData[0, frame, :, 0]
        y = skeletonData[0, frame, :, 1]
        z = skeletonData[0, frame, :, 2]

        for part in BODY_POINTS:
            xPlot = x[part]
            yPlot = y[part]
            zPlot = z[part]
            axes.plot(xPlot, zPlot, yPlot, color='g', marker='o', markerfacecolor='b')

        axes.set_xlabel('X')
        axes.set_ylabel('Z')
        axes.set_zlabel('Y')

        axes.set_facecolor('none')
        plt.pause(0.1)

    plt.ioff()
    axes.axis('off')
    plt.show()

# NTU RGB dataset contains folders of depth images, video can be made from them by this function
def prepareDepthVideo(folderName):
    imgList = glob.glob(folderName + '/*.*')
    firstImg = cv2.imread(imgList[0])
    depthVideo = cv2.VideoWriter(folderName + '_depth.avi', cv2.VideoWriter_fourcc(*'XVID'), 
                                        20.0, (int(firstImg.shape[1]),int(firstImg.shape[0])))
    for imgName in imgList:
        img = cv2.imread(imgName)
        depthVideo.write(img)
    return (folderName + '_depth.avi')

# show video with drawn skeleton
def showVideo(videoFileName, skeletonXY, mode):
    video = cv2.VideoCapture(videoFileName)
    saveVideo = cv2.VideoWriter(videoFileName.rsplit(".", 1)[0]  + '_save.avi', cv2.VideoWriter_fourcc(*'XVID'), 
                                        20.0, (int(video.get(3)),int(video.get(4))))
    skeletonXY = np.transpose(skeletonXY, (3, 1, 2, 0))
    for row in skeletonXY:
        for points in row:
            success, frame = video.read()
            if(success == True):
                # draw all points in each frame
                for point in points:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), 6)
                legPoints = []
                spinePoints = []
                armPoints = []
                for legPoint in LEG_POINTS:
                    legPoints.append([int(points[legPoint][0]), int(points[legPoint][1])])   
                for armPoint in ARM_POINTS:
                    armPoints.append([int(points[armPoint][0]), int(points[armPoint][1])])
                for spinePoint in SPINE_POINTS:
                    spinePoints.append([int(points[spinePoint][0]), int(points[spinePoint][1])])

                legPoints = np.array(legPoints, np.int32)
                armPoints = np.array(armPoints, np.int32)
                spinePoints = np.array(spinePoints, np.int32)
                # draw joint connections for each group
                cv2.polylines(frame, [legPoints], False, (255,0,0), 3)
                cv2.polylines(frame, [spinePoints], False, (255,0,0), 3)
                cv2.polylines(frame, [armPoints], False, (255,0,0), 3)
                saveVideo.write(frame)
                cv2.imshow(mode, frame)
                kk = cv2.waitKey(1) & 0xFF
                # press 'e' to exit the video
                if kk == ord('e'):
                    break


# showing the skeleton in 3D, RGB or depth
def showSkeleton(pointsData, mode, fileName = ""):
    if mode == 'xyz':
        show3DPlot(pointsData)
    elif mode == 'rgb':
        showVideo(fileName, pointsData, mode)
    elif mode == 'depth':
        showVideo(fileName, pointsData, mode)


if __name__ == '__main__':
    
    #test files names: skeleton - "S001C001P001R001A001.skeleton", rgb - "S001C001P001R001A001_rgb.avi", depthFolder = "S001C001P001R001A001/"
    #prepareDepthVideo('S001C001P001R001A001')

    pointsDepth = getPoints(readSkeleton("S001C001P001R001A001.skeleton"), 'depth')
    showSkeleton(pointsDepth, 'depth', "S001C001P001R001A001_depth.avi")
    points3D = getPoints(readSkeleton("S001C001P001R001A001.skeleton"), 'xyz')
    showSkeleton(points3D, 'xyz')
    pointsRGB = getPoints(readSkeleton("S001C001P001R001A001.skeleton"), 'rgb')
    showSkeleton(pointsRGB, 'rgb', "S001C001P001R001A001_rgb.avi")
