import numpy as np
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