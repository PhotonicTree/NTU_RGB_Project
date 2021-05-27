import os # module made for using operating system dependent functionality
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy.core.records import array

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

def read_xyz(file, max_body=2, num_joint=25):
    seq_info = readSkeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # (3,frame_nums,25 2)
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def _normal_skeleton(data):
    #  use as center joint
    center_joint = data[0, :, 0, :]

    center_jointx = np.mean(center_joint[:, 0])
    center_jointy = np.mean(center_joint[:, 1])
    center_jointz = np.mean(center_joint[:, 2])

    center = np.array([center_jointx, center_jointy, center_jointz])
    data = data - center

    return data

def _rotation(self, data, alpha=0, beta=0):
    # rotate the skeleton around x-y axis
    r_alpha = alpha * np.pi / 180
    r_beta = beta * np.pi / 180

    rx = np.array([[1, 0, 0],
                    [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                    [0, np.sin(r_alpha), np.cos(r_alpha)]]
                    )

    ry = np.array([
        [np.cos(r_beta), 0, np.sin(r_beta)],
        [0, 1, 0],
        [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
    ])

    r = ry.dot(rx)
    data = data.dot(r)

    return data

def visual_skeleton(self):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.view_init(self.init_vertical, self.init_horizon)
    plt.ion()

    data = np.transpose(self.xyz, (3, 1, 2, 0))   # Change the index value, change the index value of the x subscript to the end, and change the max_body index value to the front

    # data rotation
    if (self.x_rotation is not None) or (self.y_rotation is not None):

        if self.x_rotation > 180 or self.y_rotation > 180:
            raise Exception("rotation angle should be less than 180.")

        else:
            data = self._rotation(data, self.x_rotation, self.y_rotation)

    # data normalization
    data = self._normal_skeleton(data)

    # show every frame 3d skeleton
    for frame_idx in range(data.shape[1]):

        plt.cla()
        plt.title("Frame: {}".format(frame_idx))

        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-0.8, 0.8])

        x = data[0, frame_idx, :, 0]
        y = data[0, frame_idx, :, 1]
        z = data[0, frame_idx, :, 2]
        body = [1,2]
        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            z_plot = z[part]
            ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        if self.save_path is not None:
            save_pth = os.path.join(self.save_path, '{}.png'.format(frame_idx))
            plt.savefig(save_pth)
        print("The {} frame 3d skeleton......".format(frame_idx))

        ax.set_facecolor('none')
        plt.pause(self._pause_step)

    plt.ioff()
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    # test sample
    # self.file = file
    # self.save_path = save_path

    # #  if not os.path.exists(self.save_path):
    # #      os.mkdir(self.save_path)

    # self.xyz = self.read_xyz(self.file)
    # self.init_horizon = init_horizon
    # self.init_vertical = init_vertical

    # self.x_rotation = x_rotation
    # self.y_rotation = y_rotation

    # self._pause_step = pause_step
    print(readSkeleton("C:\\Users\\Konrad\\Documents\\Python\\Task\\S001C001P001R001A001.skeleton"))
    
