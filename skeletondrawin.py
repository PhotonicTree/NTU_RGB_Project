import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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

# getXYZ Function get as input parameter '.skeleton' file data and return XYZ Coordinates
 
def getXYZ(skletonFileData, maxSkeletons=2, numberOfJoints=25):
    xyzCoordinates = np.zeros((3, skletonFileData["numberOfFrames"], numberOfJoints, maxSkeletons))
    for countFrame, frameInfo in enumerate(skletonFileData["frameInfo"]):
        for countSkeletons, b in enumerate(frameInfo["skeletonInfo"]):
            for countJoint, joint in enumerate(b["jointInfo"]):
                if (countSkeletons < maxSkeletons and countJoint < numberOfJoints):
                    xyzCoordinates[:, countFrame, countJoint, countSkeletons] = [joint['x'], joint['y'], joint['z']]
                else:
                    pass
    return xyzCoordinates

# setCenterPoint normalization function
def setCenterPoint(skletonFileData):
    firstPoint = skletonFileData[0, :, 0, :]
    firstPointx = np.mean(firstPoint[:, 0])
    firstPointy = np.mean(firstPoint[:, 1])
    firstPointz = np.mean(firstPoint[:, 2])

    averageCenter = np.array([firstPointx, firstPointy, firstPointz])
    # reset data to have overlapping points
    skletonFileData = skletonFileData - averageCenter

    return skletonFileData


def showSkeleton(xyzCoordinates):
    pltFigure = plt.figure()
    axes = Axes3D(pltFigure)
    elevationAngle = 20
    azimuthAngle = -45
    axes.view_init(elevationAngle, azimuthAngle)
    plt.ion()

    skeletonData = np.transpose(xyzCoordinates, (3, 1, 2, 0))   # Change the index value, change the index value of the x subscript to the end, and change the max_body index value to the front

    skeletonData = setCenterPoint(skeletonData)

    # show every frame 3d skeleton
    for frame in range(skeletonData.shape[1]):
        plt.cla()
        plt.title("eluwinka")

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

if __name__ == '__main__':
    
    skeleton = getXYZ(readSkeleton("C:\\Users\\Konrad\\Documents\\Python\\Task\\S001C001P001R001A001.skeleton"))
    showSkeleton(skeleton)
    
