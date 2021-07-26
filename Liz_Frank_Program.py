from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import physvis as vis
import sys

class txt:

    def __init__(self, filename):
        self.filename = filename
        self.read_txt_file()

    def read_txt_file(filename):
        """ Read text file to find the positions
            of the screws in the foot bones.
        """

        pass

        
class experiment:

    def __init__(self, filename):
        self.foot_pair = foot_pair

        """ not really sure what to do here, basically I just
            want to pass some ID to show ownership of the foot
            in the pair """

    def graph_result():
        pass

class foot:

    def __init__(self):
        self.screw_1 = []
        self.screw_2 = []
        self.screw_3 = []
        self.screw_4 = []
        self.screw_6 = []
        self.screw_7 = []
        self.screw_8 = []
        self.screw_9 = []

class bone:

    """ Class for the 3 screws in one foot bone and their coordinates.
        Coordinates are in the fom of a list for all the time points/weights.
    """

    def __init__(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.initial_matrix = np.zeros((3,3))
        self.initialize_axes()
        self.test_matrix = np.array([1., 0., 0.],[0., 1., 0.], [0., 0., 1.])
        self.unit_vector = np.array([[1/math.sqrt(3)], [1/math.sqrt(3)], [1/math.sqrt(3)]])
        
    def mag(vector):
        return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        
    def initialize_axes(self):

        origin = np.array([self.x1, self.y1, self.z1])
        point1 = np.array([ self.x2,  self.y2,  self.z2])
        point2 = np.array([ self.x3,  self.y3,  self.z3])

        axis_1 = point1 - origin
        temp_vector = point2 - origin

        axis_2 = np.cross(axis_1, temp_vector)
        axis_3 = np.cross(axis_1, axis_2)

        self.initial_matrix[0] = axis_1/bone.mag(axis_1)
        self.initial_matrix[1] = axis_2/bone.mag(axis_2)
        self.initial_matrix[2] = axis_3/bone.mag(axis_3)

        

        
        
        sys.exit(30)

    def get_translation_vector(self, axis_1, axis_2):

        """ Uh oh, looks like we are going to have to use some quaternions here!
        """

        


    def get_rotation_vector(self):
        pass

# class visual:

class main:
    
    X1 = random.uniform(0, 5.0)
    Y1 = random.uniform(0, 5.0)
    Z1 = random.uniform(0, 5.0)
    X2 = random.uniform(0, 5.0)
    Y2 = random.uniform(0, 5.0)
    Z2 = random.uniform(0, 5.0)
    X3 = random.uniform(0, 5.0)
    Y3 = random.uniform(0, 5.0)
    Z3 = random.uniform(0, 5.0)    

    test = bone(x1 = X1, y1 = Y1, z1 = Z1, x2 = X2, y2 = Y2, z2 = Z2, x3 = X3, y3 = Y3, z3 = Z3)
    
if __name__ == "__main__":
    main()
