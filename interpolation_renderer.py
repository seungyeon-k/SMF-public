import os
import argparse
import numpy as np
import open3d as o3d
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from functions.util import gallery, render_pointcloud

if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str)
    args = parser.parse_args()

    # configuration
    run = args.run

    # random points
    data_saver = np.load(f'interpolation_results/data/{run}.npy', allow_pickle=True).item()

    # folders
    folder_temp = 'interpolation_results/temp'
    if not os.path.exists(folder_temp):
        os.makedirs(folder_temp)
    folder_png = 'interpolation_results/png'
    if not os.path.exists(folder_png):
        os.makedirs(folder_png)    
    folder_gif = 'interpolation_results/gif'
    if not os.path.exists(folder_gif):
        os.makedirs(folder_gif)   

    # ball configuration
    radius = 0.03
    resolution = 20

    # image resolution
    img_width = 640
    img_height = 900

    # make point cloud mesh
    meshes_data = []
    meshes_for_finding_pose = []
    exp_name_list = []

    for l, idx_data_ in enumerate(data_saver.keys()):
        exp_name_list.append(idx_data_)
        data_ = data_saver[idx_data_]
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

            if l == 0 and (j == 0 or j == len(points_list) - 1):
                for i in range(len(points)):
                    mesh = o3d.geometry.TriangleMesh.create_sphere(radius = radius, resolution = resolution).translate((points[i,0], points[i,1], points[i,2]))
                    mesh.paint_uniform_color([colors[i,0]/255, colors[i,1]/255, colors[i,2]/255])
                    if i == 0:
                        mesh_total = mesh
                    else:
                        mesh_total += mesh
            
                mesh_total.compute_vertex_normals()
                meshes_for_finding_pose.append(mesh_total)

    # save images
    for j in range(len(meshes_data)):
        path_image = os.path.join(folder_temp, f'temp_{j}.png')
        render_pointcloud(meshes_data[j], 
                        visualize=False, 
                        camera_config=None,
                        return_camera_config=False,
                        save_path=path_image,
                        image_size = [img_width, img_height])

    # split
    splitedSize = 20
    meshes_data_splited = [meshes_data[x:x+splitedSize] for x in range(0, len(meshes_data), splitedSize)]
    
    # append images
    images = []
    counter = 0
    for k, split_ in enumerate(meshes_data_splited):

        # figure
        fig = plt.figure()
        images_plt = []

        for j in range(len(split_)):
            im = mpimg.imread(os.path.join(folder_temp, f'temp_{counter}.png'))
            im_plt = plt.imshow(im, animated=True)
            plt.axis('off')
            images.append(im)
            images_plt.append([im_plt])
            counter += 1

        # gif file
        ani = animation.ArtistAnimation(fig, images_plt, interval=100, blit=True,
                                        repeat_delay=1000)

        file_name_video = os.path.join(folder_gif, f'{run}_{exp_name_list[k]}.gif')
        ani.save(file_name_video, dpi=300)

    # save image
    images_numpy = np.array(images)
    image_grid = gallery(images_numpy, ncols=splitedSize)
    file_name = os.path.join(folder_png, f'{run}.png')
    plt.imsave(file_name, image_grid, dpi = 1)