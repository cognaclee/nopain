import os
import logging
import numpy as np
import open3d as o3d




def reconstruct_from_pc(npoint, output_path, output_file_name, pc, output_type='mesh', normal=None,
                        reconstruct_type='PRS', central_points=None):
    # assert pc.size() == 2
    # assert pc.size(2) == 3
    # assert normal.size() == pc.size()
    print('pc', pc.shape)
    # print('central_points', central_points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # central_pcd = o3d.geometry.PointCloud()
    # central_pcd.points = o3d.utility.Vector3dVector(central_points)

    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)

    if reconstruct_type == 'BPA':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(
            [radius, radius * 2]))
        output_mesh = bpa_mesh
    elif reconstruct_type == 'PRS':
        poisson_mesh = \
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=9, width=0, scale=1.1,
                                                                      linear_fit=True, n_threads=-1)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        output_mesh = poisson_mesh.crop(bbox)

    o3d.io.write_triangle_mesh(os.path.join(output_path, output_file_name + ".obj"), output_mesh)

    output_mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.visualization.draw_geometries([output_mesh], mesh_show_wireframe=True)

    if output_type == 'mesh':
        return output_mesh
    elif output_type == 'recon_pc':
        return o3d.geometry.TriangleMesh.sample_points_uniformly(output_mesh, number_of_points=npoint)
    else:
        raise NotImplementedError


def create_logger(save_path='', file_type='', level='debug'):
    if level == 'debug':
        _level = logging.DEBUG
    else:
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger
