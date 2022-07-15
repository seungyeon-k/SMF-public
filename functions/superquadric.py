import numpy as np
import open3d as o3d
import random

class Superquadric_Object():
    
    def __init__(self, config):

        # a1        
        if config['a1']['random'] is True:
            if 'log_uniform' in config['a1'].keys() and config['a1']['log_uniform'] is True:
                a1 = random.uniform(np.log(config['a1']['min']), np.log(config['a1']['max']))
                a1 = np.exp(a1)
            else:
                a1 = random.uniform(config['a1']['min'], config['a1']['max'])
        else:
            a1 = config['a1']['value']

        # a2 
        if 'same_with_a1' in config['a2'].keys() and config['a2']['same_with_a1'] is True:
            a2 = a1
            bool_same_with_a1 = True
        else:
            if config['a2']['random'] is True:
                if 'log_uniform' in config['a2'].keys() and config['a2']['log_uniform'] is True:
                    a2 = random.uniform(np.log(config['a2']['min']), np.log(config['a2']['max']))
                    a2 = np.exp(a2)
                else:
                    a2 = random.uniform(config['a2']['min'], config['a2']['max'])
            else:
                a2 = config['a2']['value']
            bool_same_with_a1 = False

        # a3
        if 'same_with_a2' in config['a3'].keys() and config['a3']['same_with_a2'] is True:
            a3 = a2
            bool_same_with_a2 = True
        else:
            if config['a3']['random'] is True:
                if 'log_uniform' in config['a3'].keys() and config['a3']['log_uniform'] is True:
                    a3 = random.uniform(np.log(config['a3']['min']), np.log(config['a3']['max']))
                    a3 = np.exp(a3)
                else:
                    a3 = random.uniform(config['a3']['min'], config['a3']['max'])
            else:
                a3 = config['a3']['value']
            bool_same_with_a2 = False
        
        if 'sort' is True: 
            if bool_same_with_a1 is False:
                size_params = [a1, a2, a3]
                size_params.sort()
                a1 = size_params[0]
                a2 = size_params[1]
                a3 = size_params[2]
            else:
                size_params = [a1, a3]
                size_params.sort()
                a1 = size_params[0]
                a2 = size_params[0]
                a3 = size_params[1]

        # e1
        if config['e1']['random'] is True:
            e1 = random.uniform(config['e1']['min'], config['e1']['max'])
        else:
            e1 = config['e1']['value']

        # e2
        if config['e2']['random'] is True:
            e2 = random.uniform(config['e2']['min'], config['e2']['max'])
        else:
            e2 = config['e2']['value']

        # k
        if config['k']['random'] is True:
            k = random.uniform(config['k']['min'], config['k']['max'])
        else:
            k = config['k']['value']
        a1 = a1 / (1.0 - k)
        a2 = a2 / (1.0 - k)

        P1_SE3 = np.identity(4)
        P1_parameters = {"a1": a1, "a2": a2, "a3": a3, "e1": e1, "e2": e2, "k": k}
        P1 = Superquadric(SE3=P1_SE3, parameters=P1_parameters)
        self.mesh = P1.mesh

    def transform_object(self, SE3):
        self.mesh.transform(SE3)

class Superquadric():
    
    def __init__(self, SE3, parameters, color=[0.8, 0.8, 0.8]):
        
        self.SE3 = SE3
        self.parameters = parameters
        self.color = color
        self.resolution = 100

        self.mesh = self.mesh_superquadric()
        self.transform_mesh()

    def mesh_superquadric(self):
        
        assert self.SE3.shape == (4, 4)

        # parameters
        a1 = self.parameters['a1']
        a2 = self.parameters['a2']
        a3 = self.parameters['a3']
        e1 = self.parameters['e1']
        e2 = self.parameters['e2']
        k = self.parameters['k']
        R = self.SE3[0:3, 0:3]
        t = self.SE3[0:3, 3:]

        # make grids
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius = 1, resolution = self.resolution)
        vertices_numpy = np.asarray(mesh.vertices)
        eta = np.arcsin(vertices_numpy[:,2:3])
        omega = np.arctan2(vertices_numpy[:,1:2], vertices_numpy[:,0:1])

        # make new vertices
        x = a1 * fexp(np.cos(eta), e1) * fexp(np.cos(omega), e2)
        y = a2 * fexp(np.cos(eta), e1) * fexp(np.sin(omega), e2)
        z = a3 * fexp(np.sin(eta), e1)

        f_x = k / a3 * z + 1
        f_y = k / a3 * z + 1
        x = f_x * x
        y = f_y * y

        # reconstruct point matrix
        points = np.concatenate((x, y, z), axis=1)
        mesh.vertices = o3d.utility.Vector3dVector(points)

        return mesh

    def transform_mesh(self):
        self.mesh.compute_vertex_normals()
        if self.color is not None:
            self.mesh.paint_uniform_color(self.color)
        self.mesh.transform(self.SE3)

def fexp(x, p):
    return np.sign(x)*(np.abs(x)**p)