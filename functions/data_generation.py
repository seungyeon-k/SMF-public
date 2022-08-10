import numpy as np
import open3d as o3d
import os
import re
import csv
import math
import random

from functions.superquadric import Superquadric_Object

def define_SE3(R, p):
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, 3] = p
    return SE3

def generate_data(
        config, 
        num_objects=100, 
        num_pnts=1500,
        num_rots=1, 
        append=True, 
        visualize_object=False,
        visualize_pc=False, 
        dir_name=None,
        dir_key=None
    ):

    # for all objects
    for object_name in config.keys():

        # # avoid duplication
        if dir_key is None:
            object_folder = object_name
        else:
            object_folder = dir_key

        if os.path.exists(f'{dir_name}/{object_folder}') and append:
            object_numbers = []
            for filename in os.listdir(f"./{dir_name}/{object_folder}/"):
                if filename.endswith('.yml'):
                    continue
                object_numbers.append(int(re.search(r'\d+', filename).group(0)))
            obj_start_num = max(object_numbers)
            if object_numbers.count(obj_start_num) == num_rots:
                obj_start_num += 1
        else:   
            obj_start_num = 0

        # load object config
        config_obj = config[object_name]

        # iteration for the number of objects in a shape type
        for object_num in range(obj_start_num, num_objects):

            # load object
            obj = Superquadric_Object(config_obj)
            
            # visualization
            if visualize_object:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                vis.add_geometry(frame)
                vis.add_geometry(obj.mesh)
                vis.run()
                vis.destroy_window()
                    
            for view_point_num in range(num_rots):

                data = dict()

                # random orientation
                if num_rots > 1:
                    theta = np.random.uniform(-np.pi/4, np.pi/4)
                    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                    obj.transform_object(define_SE3(R, [0, 0, 0]))

                full_obj_pcd = obj.mesh.sample_points_poisson_disk(num_pnts, use_triangle_normal = True)

                # visualization
                if visualize_pc:
                    vis = o3d.visualization.Visualizer()
                    vis.create_window()
                    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                    vis.add_geometry(frame)
                    vis.add_geometry(full_obj_pcd)
                    vis.run()
                    vis.destroy_window()

                # full point cloud
                data['full_pc'] = np.concatenate((np.asarray(full_obj_pcd.points), np.asarray(full_obj_pcd.normals)), axis=1)
                
                # if directory is not specified
                if dir_name is None:
                    raise NotImplementedError

                # folder generation
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)                
                if not os.path.exists(dir_name + f'/{object_folder}'):
                    os.makedirs(dir_name + f'/{object_folder}')

                # save
                np.save(dir_name + f"/{object_folder}/{object_name}_{object_num:04}_viewpoint_{view_point_num:02}.npy", data)
                print(f"saved {object_name}_{object_num:04}_viewpoint_{view_point_num:02}")	

    return True


def save_data(default_path = '', shuffle = True, train_test_ratio = 0.6, train_val_ratio = 0.8):
    
    # load datalist
    obj_list = []
    obj_namelist = [name for name in os.listdir(default_path) if os.path.isdir(os.path.join(default_path, name)) and not name.endswith('.ipynb_checkpoints')]
    for obj_folder in obj_namelist:
        obj_path = default_path + '/' + obj_folder
        file_list = os.listdir(obj_path)
        file_list.sort()

        # object categoralize
        obj_category = []
        obj_index = 0
        while(True):
            prefix = obj_folder + '_' + str(obj_index)
            file_list_category = [file for file in file_list if file.startswith(prefix)]
            if not file_list_category:
                break

            else:
                if shuffle == True:
                    random.shuffle(file_list_category)
                obj_category.append(file_list_category)
                obj_index += 1

        obj_list.append(obj_category)

    # save training data
    csv_path = default_path + '/' + 'train_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(0, math.floor(len(obj) * train_test_ratio * train_val_ratio)):
                    writer.writerow([obj[obj_index]]) 

    # save validation data
    csv_path = default_path + '/' + 'validation_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(math.floor(len(obj) * train_test_ratio * train_val_ratio), math.floor(len(obj) * train_test_ratio)):
                    writer.writerow([obj[obj_index]]) 

    # save test data
    csv_path = default_path + '/' + 'test_datalist.csv'
    with open(csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile, 
                            delimiter=',',
                            quotechar='"', 
                            quoting=csv.QUOTE_MINIMAL)
        for category in obj_list:
            for obj in category:
                for obj_index in range(math.floor(len(obj) * train_test_ratio), len(obj)):
                    writer.writerow([obj[obj_index]])   