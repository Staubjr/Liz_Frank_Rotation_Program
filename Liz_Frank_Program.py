from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import physvis as vis
import sys

class txt:

    """ Reads text files
    """

    def __init__(self, filename):
        self.filename = filename
        self.read_txt_file()

    def read_txt_file(filename):
        """ Read text file to find the positions
            of the screws in the foot bones.
        """

        pass

        
class experiment:

    """ Class for a single experiment that holds the objects in the foot class
    """

    all_feet = []
    
    def __init__(self, filename):
        self.foot_pair = foot_pair

        """ not really sure what to do here, basically I just
            want to pass some ID to show ownership of the foot
            in the pair """

    def graph_result():
        pass

class foot:

    all_bones = []

    """ Class for a single foot that holds the objects of the bone class.  Also, let's
        write this so that can tell if the foot is a left or a right as well, just purely
        based on the ID.  Not sure if I'll actually do something with this or raise an
        error if there is an issue...
    """
    
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

    """ Class that stores the objects of the screw class.  Each bone object has three screw
        objects in it.
    """
    all_screws = []
    
    def __init__(self, x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self.origin_screw = screw(x = x1, y = y1, z = z1, identity = "origin")
        self.screw_2 = screw(x = x2, y = y2, z = z2, identity = "screw_2")
        self.screw_3 = screw(x = x3, y = y3, z = z3, identity = "screw_3")
        self.local_axes_matrix = np.zeros((3,3))
        self.global_axes_matrix = np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])
        self.unit_vector = np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
        self.initialize_axes()
        self.translate_screws()
        self.get_rotation_matrix(self.global_axes_matrix, self.local_axes_matrix)
        self.visualize_bone()
        
    def mag(vector):
        return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        
    def initialize_axes(self):

        origin = self.origin_screw.pos
        point1 = self.screw_2.pos
        point2 = self.screw_3.pos

        axis_1 = self.screw_2.pos - self.origin_screw.pos
        temp_vector = self.screw_3.pos - self.origin_screw.pos

        axis_2 = np.cross(axis_1, temp_vector)
        axis_3 = np.cross(axis_1, axis_2)
        axis_3 -= 2*axis_3

        self.local_axes_matrix[0] = axis_1/bone.mag(axis_1)
        self.local_axes_matrix[1] = axis_2/bone.mag(axis_2)
        self.local_axes_matrix[2] = axis_3/bone.mag(axis_3)

    def translate_screws(self):
        
        translation_vector = -1 * self.origin_screw.pos
        self.origin_screw.pos += translation_vector
        self.screw_2.pos += translation_vector
        self.screw_3.pos += translation_vector
        
    def get_rotation_matrix(self, global_axes, local_axes):

        """ Note that 1, 2, 3 corresponds to x, y, z both globally and locally """

        axis_1 = global_axes[0]
        axis_2 = global_axes[1]
        axis_3 = global_axes[2]
        axis_4 = local_axes[0]
        axis_5 = local_axes[2]
        axis_6 = local_axes[1]

        axis_4_hat = axis_4/bone.mag(axis_4)
        axis_5_hat = axis_5/bone.mag(axis_5)
        axis_6_hat = axis_6/bone.mag(axis_6)


#### New euler angle method with the following rotations: z --> x' --> z'

        axis_1_prime = np.cross(axis_3, axis_6)
        axis_1_prime /= bone.mag(axis_1_prime)

        # if axis_3[0] == axis_6_hat[0] and axis_3[1] == axis_3[1] and axis_3[2] == axis_3[2]:
        #     axis_1_prime = np.array([0., 0., 0.])
        
        phi = math.acos( axis_1.dot(axis_1_prime) / ( bone.mag(axis_1) * bone.mag(axis_1_prime) ) )

        if axis_1_prime[1] < 0.0:
            phi = 2*math.pi - phi

        rotation_matrix_axis_3 = np.array([ [ math.cos(phi), math.sin(phi), 0.],
                                            [-math.sin(phi), math.cos(phi), 0.],
                                            [ 0.           , 0.           , 1.] ] )

        theta = math.acos( axis_3.dot(axis_6) / ( bone.mag(axis_3) * bone.mag(axis_6) ) )

        psi = math.acos( axis_1_prime.dot(axis_4) / (bone.mag(axis_1_prime) * bone.mag(axis_4) ) )                

        # rotation matrix about the z-axis
        
        rotation_matrix_1 = np.array([ [ math.cos(phi), math.sin(phi), 0.],
                                       [-math.sin(phi), math.cos(phi), 0.],
                                       [0.            , 0.           , 1.] ] )

        # rotation matrix about the y' / y" 

        rotation_matrix_2 = np.array([ [1.0, 0.              , 0.               ],
                                       [0., math.cos(theta)  ,math.sin(theta)  ],
                                       [0.,-math.sin(theta) , math.cos(theta)  ] ] )
        
        # rotation matrix about the z'-axis

        rotation_matrix_3 = np.array([ [math.cos(psi), math.sin(psi), 0.],
                                       [-math.sin(psi), math.cos(psi), 0.],
                                       [0.            , 0.           , 1.] ] )

        euler_rotation_matrix = rotation_matrix_3.dot(rotation_matrix_2.dot(rotation_matrix_1))

        # euler_rotation_matrix  = np.array([[math.cos(psi)*math.cos(phi) - math.cos(theta)*math.sin(psi)*math.sin(phi), -math.sin(psi)*math.cos(phi) - math.cos(theta)*math.sin(phi)*math.cos(psi),  math.sin(theta)*math.sin(phi)],
        #                                   [math.cos(psi)*math.sin(phi) + math.cos(theta)*math.cos(phi)*math.sin(psi), -math.sin(phi)*math.sin(psi) + math.cos(theta)*math.cos(phi)*math.cos(psi), -math.sin(theta)*math.cos(phi)],
        #                                   [math.sin(theta)*math.sin(psi)                                            , math.sin(theta)*math.cos(psi)                                             ,  math.cos(theta)              ] ] )

        axis_1_prime_check = axis_1.dot(rotation_matrix_1)        
        axis_1_prime_prime_check = axis_1.dot(rotation_matrix_2.dot(rotation_matrix_1))
        axis_1_prime_prime_prime_check = axis_1.dot(rotation_matrix_3.dot(rotation_matrix_2.dot(rotation_matrix_1)))

        # print(axis_1_prime_check)
        # print(axis_1_prime_prime_check)
        # print(axis_1_prime_prime_prime_check)
        # print(phi)
        # print(theta)
        print(psi)

        vis.arrow( pos = (0., 0., 0.), axis = axis_1_prime_check, color = (1.0, 0., 1.0) )
        vis.arrow( pos = (0., 0., 0.), axis = axis_1_prime_prime_check, color = (1.0, 1.0, 0.) )
        vis.arrow( pos = (0., 0., 0.), axis = axis_1_prime_prime_prime_check, color = (0., 1.0, 1.0) )
        
        
    def visualize_bone(self):

        self.origin_screw.visualize_screw()
        self.screw_2.visualize_screw()
        self.screw_3.visualize_screw()
        visual(self)

class screw:

    """ Each screw object has an <x, y, z> coordinate and an identity.
    """

    def __init__(self, x, y, z, identity):
        self.x = x
        self.y = y
        self.z = z
        self.pos = np.array([self.x, self.y, self.z])
        self.identity = identity
        self.bone = []

    def visualize_screw(self):
        visual_screw = visual(self)
        
class visual:

    """ Creates the screws and axes (both the global coordinate system and the local
        coordinate system.  Also, allows for the implementation of the rotation to 
        make sure the rotation matrix is working as it should.
    """

    all_visual_screws = []

    all_visual_axes = []

    screw_color = { "origin"   : (255., 0., 0.),
                    "screw_2"   : (0., 255., 0.),
                    "screw_3" : (0., 0., 255.)  }

    arrow_color = ( 255.0/255, 222.0/255, 0.)

    def __init__(self, object):
        
        if object.__class__.__name__ == 'screw':
            self.screw_object = object
            self.visual = vis.sphere(pos = object.pos, radius = 0.125, color = visual.screw_color[object.identity])
            visual.all_visual_screws.append(self)

        if object.__class__.__name__ == 'bone':
            self.axes_object = object
            self.visual_axis_1 = vis.arrow(pos = object.origin_screw.pos, axis = object.local_axes_matrix[0], color = (1., 0., 0.))
            self.visual_axis_2 = vis.arrow(pos = object.origin_screw.pos, axis = object.local_axes_matrix[2], color = (0., 1., 0.))
            self.visual_axis_3 = vis.arrow(pos = object.origin_screw.pos, axis = object.local_axes_matrix[1], color = (0., 0., 1.))
            visual.all_visual_axes.append(self)

######################################################################################################################################            

def main():

    """ Main function to run program.
        First, let's still use the random function to make one screw close to the origin, one screw
        close to the x-axis, and one screw close to the y-axis.  This way we should get a local coordinate system
        that is close to the actual global coordinate system.

    """
    
# Origin Screw
    
    X1 = random.uniform(0, 0.25)
    Y1 = random.uniform(0, 0.25)
    Z1 = random.uniform(0, 0.25)
    
# X-axis Screw
    
    X2 = random.uniform(3.0, 5.0)
    Y2 = random.uniform(0, 2.0)
    Z2 = random.uniform(0, 2.0)

# Y-axis Screw
        
    X3 = random.uniform(0, 2.0)
    Y3 = random.uniform(3.0, 5.0)
    Z3 = random.uniform(0, 2.0)    

    test = bone(x1 = X1, y1 = Y1, z1 = Z1, x2 = X2, y2 = Y2, z2 = Z2, x3 = X3, y3 = Y3, z3 = Z3)

    t = 0
    dt = 1E-3

    vis.xaxis()
    vis.yaxis()
    vis.zaxis()

    while t <= 100:
        
        vis.rate(30)
        
    
if __name__ == "__main__":
    main()

#### Old euler angle method with the following rotations: z --> y' --> z'
        
        # axis_1_prime = np.array([axis_6[0], axis_6[1], 0.])
        # phi = math.acos( axis_1.dot(axis_1_prime) / (bone.mag(axis_1) * bone.mag(axis_1_prime)) )

        # axis_3_rotation_matrix = np.array([[ math.cos(phi), 1*math.sin(phi) , 0.],
        #                                          [-math.sin(phi), math.cos(phi)   , 0.],
        #                                          [ 0.           , 0.              , 1.]])

        # axis_2_prime = axis_2.dot(axis_3_rotation_matrix)
        
        # theta = math.acos( axis_3.dot(axis_6) / ( bone.mag(axis_3) * bone.mag(axis_6) ) )
        
        # psi = math.acos( axis_2_prime.dot(axis_5) / (bone.mag(axis_2_prime) * bone.mag(axis_5) ) )

    
        # euler_rotation_matrix = np.array([ [math.cos(psi)*math.cos(theta)*math.cos(phi) - math.sin(psi)*math.sin(phi)  , -math.cos(psi)*math.sin(phi) - math.sin(psi)*math.cos(theta)*math.cos(phi), math.sin(theta)*math.cos(phi)],
        #                                    [math.cos(psi)*math.cos(theta)*math.sin(phi) + math.sin(psi)*math.cos(phi)  , math.cos(psi)*math.cos(phi) - math.sin(psi)*math.cos(theta)*math.sin(phi) , math.sin(theta)*math.sin(phi)],
        #                                    [-math.cos(psi)*math.sin(theta)                                             , math.sin(psi)*math.sin(theta)                                             , math.cos(theta)              ] ])
        # This is Dr. Knop's version that is not correct

        # euler_rotation_matrix_check = np.array([ [math.cos(psi)*math.cos(phi) - math.cos(theta)*math.sin(phi)*math.sin(psi) , math.cos(psi)*math.sin(phi) + math.cos(theta)*math.cos(phi)*math.sin(psi) , math.sin(psi)*math.sin(theta)],
        #                                    [-math.sin(psi)*math.cos(phi) - math.cos(theta)*math.sin(phi)*math.cos(psi), -math.sin(psi)*math.sin(phi) + math.cos(theta)*math.cos(phi)*math.cos(psi), math.cos(psi)*math.sin(theta)],
        #                                    [math.sin(theta)*math.sin(phi)                                             , -math.sin(theta)*math.cos(phi)                                            , math.cos(theta)              ] ])
        # this is the one based on lines 6-14, this appears to flip the x and y

        # euler_rotation_matrix = np.array([ [-math.sin(psi)*math.sin(phi) + math.cos(theta)*math.cos(phi)*math.cos(psi), math.sin(psi)*math.cos(phi) + math.cos(theta)*math.sin(phi)*math.cos(psi), -math.cos(psi)*math.sin(theta)],
        #                                    [-math.cos(psi)*math.sin(phi) - math.cos(theta)*math.cos(phi)*math.sin(psi), math.cos(psi)*math.cos(phi) - math.cos(theta)*math.sin(phi)*math.sin(psi), math.sin(psi)*math.sin(theta) ],
        #                                    [math.sin(theta)*math.cos(phi)                                             , math.sin(theta)*math.sin(phi)                                            , math.cos(theta)               ] ] )
        # this is the one based on lines 39-47, this sometimes flips the y in sign, but all magnitudes are correct
