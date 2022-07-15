import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from copy import deepcopy
from functions.lie_alg import define_SE3
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def render_pointcloud(X, 
                    visualize=True, 
                    camera_config=None,
                    return_camera_config=True,
                    save_path=None,
                    image_size=[600, 960]):

    # define ground plane
    a = 10.0
    plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
    plane.paint_uniform_color([1.0, 1.0, 1.0])
    plane.translate([-a/2, -a/2, -0.45])
    plane.compute_vertex_normals()
    mat_plane = rendering.Material()
    mat_plane.shader = 'defaultLit'
    mat_plane.base_color = [1.0, 1.0, 1.0, 4.0]

    # object material
    mat = rendering.Material()
    mat.shader = 'defaultLit'

    # set window
    if visualize:
        gui.Application.instance.initialize()
        window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
        widget = gui.SceneWidget()
        widget.scene = rendering.Open3DScene(window.renderer)
        window.add_child(widget) 
    else:
        gui.Application.instance.initialize()
        widget = o3d.visualization.rendering.OffscreenRenderer(image_size[0], image_size[1])

    # camera view point selection
    # widget.scene.camera.look_at([0,0,0], [1,1,1], [0,0,1])
    # widget.setup_camera(60.0, [0, 0, 0], [0.0, 1.0, 0.7], [0, 0, 1])
    widget.setup_camera(60.0, [0, 0, 0], [0.5, 0.86602540378, 0.7], [0, 0, 1])
    if camera_config is not None:
        widget.scene.camera.copy_from(camera_config)

    # add geometries and lighting
    widget.scene.add_geometry('mesh', X, mat)
    widget.scene.add_geometry('plane', plane, mat_plane)
    widget.scene.set_lighting(widget.scene.LightingProfile.DARK_SHADOWS, (0.3, -0.3, -0.9))
    widget.scene.set_background([1.0, 1.0, 1.0, 3.0], image=None)

    if visualize:
        while gui.Application.instance.run_one_tick():
            camera_config_current = widget.scene.camera

        if return_camera_config:
            return camera_config_current
    else:
        img_o3d = widget.render_to_image()
        if save_path is not None:
            o3d.io.write_image(save_path, img_o3d, 9)
            print(f'image saved with name : {save_path}')
        else:
            o3d.io.write_image('temp.png', img_o3d, 9)
            print('image saved with name : temp.png')
 
def finding_camera_pose(X1, X2, image_size):
    
    while True:
        # first image
        camera_config_X1 = render_pointcloud(X1, 
                                        visualize=True, 
                                        camera_config=None,
                                        return_camera_config=True,
                                        image_size=image_size)

        # second image
        render_pointcloud(X2, 
                        visualize=False, 
                        camera_config=camera_config_X1,
                        return_camera_config=False,
                        image_size=image_size)

        img = mpimg.imread('temp.png')
        plt.imshow(img)
        plt.show()
        
        # get answer
        while True:
            answer = input('Proceeding? (y/n)')
            if answer == 'y':
                terminate = True
                break
            elif answer == 'n':
                terminate = False
                break
            else:
                print('invalid keys!')

        # terminate
        if terminate:
            break
        else:
            del camera_config_X1

    return camera_config_X1

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