import numpy as np
from datetime import datetime

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib.patches import Ellipse, Rectangle, Polygon

def label_to_color(label):
    
    n_points = label.shape[0]
    max_label = np.max(label)
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((10, 3))
    rgb[0, :] = [253, 134, 18] # box, orange
    # rgb[0, :] = [179, 173, 151] # box, brown
    rgb[1, :] = [106, 194, 217] # cylinder, blue
    rgb[2, :] = [111, 146, 110] # cone, green
    rgb[3, :] = [153, 0, 17] # ellipsoid, red
    # rgb[4, :] = [219, 226, 233] # interpolates, grey
    rgb[4, :] = [155, 155, 155] # interpolates, grey
    rgb[5, :] = [179, 173, 151] # brown
    rgb[6, :] = [245, 228, 0] # yellow
    rgb[7, :] = [255, 0, 0] 
    rgb[8, :] = [0, 255, 0] 
    rgb[9, :] = [0, 0, 255]  

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]

    return color

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
                    camera_position=[0.0, 1.0, 0.7],
                    image_size=[600, 960]):

    # define ground plane
    a = 10.0
    plane = o3d.geometry.TriangleMesh.create_box(width=a, depth=0.05, height=a)
    plane.paint_uniform_color([1.0, 1.0, 1.0])
    plane.translate([-a/2, -a/2, -0.6])
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
    widget.setup_camera(60.0, [0, 0, 0], camera_position, [0, 0, 1])
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

def figure_to_array(fig):
    
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def rectangle_scatter(size, center, color):

    return Rectangle(xy=(center[0]-size[0]/2, center[1]-size[1]/2) ,width=size[0], height=size[1], facecolor=color)

def triangle_scatter(size, center, color):
    
    return Polygon(((center[0], center[1] + size[1]/2), (center[0] - size[0]/2, center[1] - size[1]/2), (center[0] + size[0]/2, center[1] - size[1]/2)), fc=color)