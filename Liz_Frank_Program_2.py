from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import random
import matplotlib.pyplot as plt
#import physvis as vis
import sys

class txt:
    
    """ Reads text files for left and right foot of individual cadaver
    """

    def __init__(self, foot_file):
        patient_name = foot_file.strip('.csv')
        self.patient_name = str(patient_name)
        self.read_txt_file(foot_file)

    def read_txt_file(self, filename):
        """ Read text file to find the positions
            of the screws in the foot bones.
        """
        Fixations = []
        MT_Proximal_xs = []
        MT_Proximal_ys = []
        MT_Proximal_zs = []
        MT_Distal_xs = []
        MT_Distal_ys = []
        MT_Distal_zs = []
        MT_3rd_xs = []
        MT_3rd_ys = []
        MT_3rd_zs = []
        
        MC_Proximal_xs = []
        MC_Proximal_ys = []
        MC_Proximal_zs = []
        MC_Distal_xs = []
        MC_Distal_ys = []
        MC_Distal_zs = []
        MC_3rd_xs = []
        MC_3rd_ys = []
        MC_3rd_zs = []

        IC_Proximal_xs = []
        IC_Proximal_ys = []
        IC_Proximal_zs = []
        IC_Distal_xs = []
        IC_Distal_ys = []
        IC_Distal_zs = []
        IC_3rd_xs = []
        IC_3rd_ys = []
        IC_3rd_zs = []
               
        file = open(filename, 'r')
        lines = file.readlines()
        total_line_number = len(lines)

        stop = False
        
        for line_number in range(total_line_number):
            if stop == False:
            
                line = lines[line_number].strip(",,,,\n")
                line = line.split(",")

                header = False
                
                if len(line) == 1:

                    potential_header = str(line[0])
                
                    for index in range(0,len(potential_header)):
                
                        if potential_header[index] == str("%"):
                            header = True

                if header == True:
                            
                    check = lines[line_number + 1]
                    check = check.strip(",,,,\n")
                    check = check.split(",")
                        
                    if len(check) == 1:
                        stop = True
                    
                if header == True and stop == False:

                    Fixations.append(potential_header)
                    
                    line = lines[line_number+1].split(',')

                    MT_Proximal_xs.append(float(line[2]))
                    MT_Proximal_ys.append(float(line[3]))
                    MT_Proximal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+2].split(',')
                    
                    MT_Distal_xs.append(float(line[2]))
                    MT_Distal_ys.append(float(line[3]))
                    MT_Distal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+7].split(',')
                    
                    MT_3rd_xs.append(float(line[2]))
                    MT_3rd_ys.append(float(line[3]))
                    MT_3rd_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+3].split(',')
                    
                    MC_Proximal_xs.append(float(line[2]))
                    MC_Proximal_ys.append(float(line[3]))
                    MC_Proximal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+4].split(',')
                    
                    MC_Distal_xs.append(float(line[2]))
                    MC_Distal_ys.append(float(line[3]))
                    MC_Distal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+8].split(',')
                    
                    MC_3rd_xs.append(float(line[2]))
                    MC_3rd_ys.append(float(line[3]))
                    MC_3rd_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+5].split(',')
                    
                    IC_Proximal_xs.append(float(line[2]))
                    IC_Proximal_ys.append(float(line[3]))
                    IC_Proximal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+6].split(',')
                    
                    IC_Distal_xs.append(float(line[2]))
                    IC_Distal_ys.append(float(line[3]))
                    IC_Distal_zs.append(float(line[4]))                    
                    
                    line = lines[line_number+9].split(',')
                    
                    z_value = line[4].strip('\n')
                    
                    IC_3rd_xs.append(float(line[3]))
                    IC_3rd_ys.append(float(line[4]))
                    IC_3rd_zs.append(float(z_value))
                    
        MT_Proximal = screw(Fixations, MT_Proximal_xs, MT_Proximal_ys, MT_Proximal_zs, identity = "MT_Proximal")
        MT_Distal = screw(Fixations, MT_Distal_xs, MT_Distal_ys, MT_Distal_zs, identity = "MT_Distal")
        MT_3rd = screw(Fixations, MT_3rd_xs, MT_3rd_ys, MT_3rd_zs, identity = "MT_3rd")
        MC_Proximal = screw(Fixations, MC_Proximal_xs, MC_Proximal_ys, MC_Proximal_zs, identity = "MC_Proximal")
        MC_Distal = screw(Fixations, MC_Distal_xs, MC_Distal_ys, MC_Distal_zs, identity = "MC_Distal")
        MC_3rd = screw(Fixations, MC_3rd_xs, MC_3rd_ys, MC_3rd_zs, identity = "MC_3rd")
        IC_Proximal = screw(Fixations, IC_Proximal_xs, IC_Proximal_ys, IC_Proximal_zs, identity = "IC_Proximal")
        IC_Distal = screw(Fixations, IC_Distal_xs, IC_Distal_ys, IC_Distal_zs, identity = "IC_Distal")
        IC_3rd = screw(Fixations, IC_3rd_xs, IC_3rd_ys, IC_3rd_zs, identity = "IC_3rd")

        MT = bone(MT_Proximal, MT_Distal, MT_3rd, Fixations, "MT")
        MC = bone(MC_Proximal, MC_Distal, MC_3rd, Fixations, "MC")
        IC = bone(IC_Proximal, IC_Distal, IC_3rd, Fixations, "IC")

        my_experiment = foot(MT, MC, IC, Fixations)
        self.make_txt_file(my_experiment)

    def make_txt_file(self, experiment):

        file = open('{}_Matrices.txt'.format(str(self.patient_name)), 'w')
        file_name = str('{}_Matrices.txt'.format(self.patient_name))

        for index in range(len(experiment.fixations)):
            file.write(experiment.fixations[index])
            file.write("\n")
            file.write(experiment.bones[0].bone_identity)
            file.write("\n")
            file.write(np.array2string(experiment.bones[0].matrices[index]))
            file.write("\n")
            file.write(experiment.bones[1].bone_identity)
            file.write("\n")
            file.write(np.array2string(experiment.bones[1].matrices[index]))
            file.write("\n")
            file.write(experiment.bones[2].bone_identity)
            file.write("\n")
            file.write(np.array2string(experiment.bones[2].matrices[index]))
            file.write("\n\n")
                    
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

    """ Class for a single foot that holds the objects of the bone class.  Also, let's
        write this so that can tell if the foot is a left or a right as well, just purely
        based on the ID.  Not sure if I'll actually do something with this or raise an
        error if there is an issue...
    """
    
    def __init__(self, bone_1, bone_2, bone_3, fixations):
        self.bones = [bone_1, bone_2, bone_3]
        self.fixations = fixations


class bone:

    """ Class that stores the objects of the screw class.  Each bone object has three screw
        objects in it.

    origin_screw = 2MT
    screw_2      = MC
    screw_3      = 

    """
    all_screws = []
    matrices = []

    def __init__(self, screw_1, screw_2, screw_3, fixations, bone_identity):
        self.origin_screw = screw_1
        self.screw_2 = screw_2
        self.screw_3 = screw_3
        self.fixations = fixations
        self.number_of_fixations = len(fixations)
        self.bone_identity = bone_identity
        self.translation_vector = np.array([0., 0., 0.])
        self.local_axes_matrix = np.zeros((3,3))
        self.global_axes_matrix = np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])
        self.unit_vector = np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])

        for index in range(self.number_of_fixations):
            self.origin_screw.pos = np.array([self.origin_screw.xs[index], self.origin_screw.ys[index], self.origin_screw.zs[index]])  
            self.screw_2.pos = np.array([self.screw_2.xs[index], self.screw_2.ys[index], self.screw_2.zs[index]])  
            self.screw_3.pos = np.array([self.screw_3.xs[index], self.screw_3.ys[index], self.screw_3.zs[index]])  
            
            self.initialize_axes()
            self.translate_screws()
            self.get_rotation_matrix(self.global_axes_matrix, self.local_axes_matrix)
        # self.visualize_bone()
        
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
        self.translation_vector = translation_vector
        
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

        final_matrix = np.zeros((4,4))

        final_matrix[0][0] = euler_rotation_matrix[0][0]
        final_matrix[0][1] = euler_rotation_matrix[0][1]
        final_matrix[0][2] = euler_rotation_matrix[0][2]
        final_matrix[1][0] = euler_rotation_matrix[1][0]
        final_matrix[1][1] = euler_rotation_matrix[1][1]
        final_matrix[1][2] = euler_rotation_matrix[1][2]
        final_matrix[2][0] = euler_rotation_matrix[2][0]
        final_matrix[2][1] = euler_rotation_matrix[2][1]
        final_matrix[2][2] = euler_rotation_matrix[2][2]
        
        final_matrix[0][3] = self.translation_vector[0]
        final_matrix[1][3] = self.translation_vector[1]
        final_matrix[2][3] = self.translation_vector[2]
        final_matrix[3][3] = float(1.0)

        bone.matrices.append(final_matrix)


        # axis_1_prime_check = axis_1.dot(rotation_matrix_1)        
        # axis_1_prime_prime_check = axis_1.dot(rotation_matrix_2.dot(rotation_matrix_1))
        # axis_1_prime_prime_prime_check = axis_1.dot(rotation_matrix_3.dot(rotation_matrix_2.dot(rotation_matrix_1)))

        # print(axis_1_prime_check)
        # print(axis_1_prime_prime_check)
        # print(axis_1_prime_prime_prime_check)
        # print(phi)
        # print(theta)
        # print(psi)

        # check_1 = axis_1.dot(euler_rotation_matrix)
        # check_2 = axis_2.dot(euler_rotation_matrix)
        # check_3 = axis_3.dot(euler_rotation_matrix)
        
    def visualize_bone(self):

        self.origin_screw.visualize_screw()
        self.screw_2.visualize_screw()
        self.screw_3.visualize_screw()
        visual(self)

class screw:

    """ Each screw object has an <x, y, z> coordinate and an identity.
    """

    def __init__(self, fixations, xs, ys, zs, identity):
        self.fixations = fixations
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.positions = []
        
        for index in range(len(self.fixations)):
            self.positions.append(np.array([self.xs[index], self.ys[index], self.zs[index]]))

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

    foot_file = sys.argv[1]
                                  
    txt(foot_file)
        
    
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

#### Old physvis stuff in the main function ####
# # Origin Screw
    
#     X1 = random.uniform(0, 0.25)
#     Y1 = random.uniform(0, 0.25)
#     Z1 = random.uniform(0, 0.25)
    
# # X-axis Screw
    
#     X2 = random.uniform(3.0, 5.0)
#     Y2 = random.uniform(0, 2.0)
#     Z2 = random.uniform(0, 2.0)

# # Y-axis Screw
        
#     X3 = random.uniform(0, 2.0)
#     Y3 = random.uniform(3.0, 5.0)
#     Z3 = random.uniform(0, 2.0)    

#     test = bone(x1 = X1, y1 = Y1, z1 = Z1, x2 = X2, y2 = Y2, z2 = Z2, x3 = X3, y3 = Y3, z3 = Z3)

#     # t = 0
#     # dt = 1E-3

#     # vis.xaxis()
#     # vis.yaxis()
#     # vis.zaxis()

#     # while t <= 100:
        
#     #     vis.rate(30)

# Git Hub Token: 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
