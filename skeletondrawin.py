import numpy as np

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
        for t in range(skeletonFileData["numberOfFrames"]):
            frameInfo = dict()
            frameInfo["numberOfSkeletons"] = int(file.readline())
            frameInfo["skeletonInfo"] = []
            for m in range(frameInfo["numberOfSkeletons"]):
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

def getXYZ(skletonFileData, maxSkeletons=2, numberOfJoints=25):
    data = np.zeros((3, skletonFileData["numberOfFrames"], numberOfJoints, maxSkeletons))
    for countFrame, frameInfo in enumerate(skletonFileData["frameInfo"]):
        for countSkeletons, b in enumerate(frameInfo["skeletonInfo"]):
            for countJoint, joint in enumerate(b["jointInfo"]):
                if (countSkeletons < maxSkeletons and countJoint < numberOfJoints):
                    data[:, countFrame, countJoint, countSkeletons] = [joint['x'], joint['y'], joint['z']]
                else:
                    pass
    return data

if __name__ == '__main__':
    print(np.zeros((3,70,25,2)))
    #print(readSkeleton("C:\\Users\\Konrad\\Documents\\Python\\Task\\S001C001P001R001A001.skeleton"))
    
