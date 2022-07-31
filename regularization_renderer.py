import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from functions.util import gallery, render_pointcloud

if __name__ == '__main__':

    # random points
    data_saver = np.load('figures/figure_2/20210906-1025.npy', allow_pickle=True).item()
    # data = np.load('figures/figure_2/20210810-2157.npy', allow_pickle=True).item()

    # save file name
    # file_name_prefix = 'figures/figure_1_png/' + str(datetime.now().strftime('%Y%m%d-%H%M'))
    file_name_prefix_temp = 'figures/temp/'
    file_name_prefix = 'figures/figure_2_png/' + str(datetime.now().strftime('%Y%m%d-%H%M'))
    file_name_prefix_video = 'figures/figure_2_gif/' + str(datetime.now().strftime('%Y%m%d-%H%M'))

    # ball configuration
    radius = 0.03
    resolution = 20

    # image resolution
    img_width = 500
    img_height = 600

    # reg_type
    for k, idx_data in enumerate(data_saver.keys()):
        data = data_saver[idx_data]
        meshes_data = []

        data_keys = list(data.keys())
        data_keys.sort()
        
        for idx_data_ in data_keys:
            data_ = data[idx_data_]
            points_list = data_['pc']
            colors_list = data_['color']

            for j in range(len(points_list)):
                points = points_list[j].transpose()
                colors = colors_list[j].transpose()

                for i in range(len(points)):
                    mesh = o3d.geometry.TriangleMesh.create_sphere(radius = radius, resolution = resolution).translate((points[i,0], points[i,1], points[i,2]))
                    mesh.paint_uniform_color([colors[i,0]/255, colors[i,1]/255, colors[i,2]/255])
                    if i == 0:
                        mesh_total = mesh
                    else:
                        mesh_total += mesh
                
                mesh_total.compute_vertex_normals()
                meshes_data.append(mesh_total)
        
        # save images
        for j in range(len(meshes_data)):
            path_image = file_name_prefix_temp + f'temp_{j}.png'
            render_pointcloud(meshes_data[j], 
                            visualize=False, 
                            camera_config=None,
                            return_camera_config=False,
                            save_path=path_image,
                            camera_position=[0.5, 0.86602540378, 0.7],
                            image_size = [img_width, img_height])

        # split
        splitedSize = 10
        meshes_data_splited = [meshes_data[x:x+splitedSize] for x in range(0, len(meshes_data), splitedSize)]
        
        # append images
        images = []
        counter = 0
        for k, split_ in enumerate(meshes_data_splited):

            for j in range(len(split_)):
                im = mpimg.imread(file_name_prefix_temp + f'temp_{counter}.png')
                images.append(im)
                counter += 1

        # save image
        images_numpy = np.array(images)
        image_grid = gallery(images_numpy, ncols=splitedSize)
        file_name = file_name_prefix + f'_interp_{idx_data}.png'
        plt.imsave(file_name, image_grid, dpi = 1)