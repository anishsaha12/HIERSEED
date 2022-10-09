import numpy as np
import itertools
import matplotlib.pyplot as plt

# Reference: Largest Empty Circle - https://www.youtube.com/watch?v=dijZkOCNMo0

def voronoi_cluster(centers, dataset, std_tolerance_multiplier = 10, verbose=False):
    # centers: the embeddings of the sub-topics
    # dataset: the datapoints assigned to main-topic
    points = centers
    neighbors = dataset
    
    if len(points)<=1:
        return None,None,None
    elif len(points)==2:
        indices_set_2 = np.array([(0,1)])

        neighbor_distances_set_2 = []
        for neighbor in neighbors:
            center_distances = []
            for index_set in indices_set_2:
                dist = (
                    np.sqrt(np.sum((neighbor-points[index_set[0]])**2)),
                    np.sqrt(np.sum((neighbor-points[index_set[1]])**2))
                )
                center_distances.append(np.std(dist))
            neighbor_distances_set_2.append(center_distances)
        neighbor_distances_set_2 = np.array(neighbor_distances_set_2)  

        neighbor_center_distances = neighbor_distances_set_2.min(axis=1)
        neighbor_center_indices = neighbor_distances_set_2.argmin(axis=1)

        if verbose:
            print('Minimum Neighbor-Center Distance:',neighbor_center_distances.min())

        dist_std_tolerance = neighbor_center_distances.min()*std_tolerance_multiplier
        if verbose:
            print('Number of Vertices identified:',len(neighbor_center_indices[(neighbor_center_distances <= dist_std_tolerance)]))

        voronoi_point_indices = indices_set_2[neighbor_center_indices[(neighbor_center_distances <= dist_std_tolerance)]]
        voronoi_points = points[voronoi_point_indices]
        voronoi_vertices = neighbors[(neighbor_center_distances <= dist_std_tolerance)]

        vertex_radius = []
        for i in range(len(voronoi_vertices)):
            dist = (
                np.sqrt(np.sum((voronoi_vertices[i]-voronoi_points[i][0])**2)),
                np.sqrt(np.sum((voronoi_vertices[i]-voronoi_points[i][1])**2))
            )
            vertex_radius.append((i, np.mean(dist)))
        return voronoi_vertices, voronoi_points, vertex_radius
    else:
        indices = list(range(len(points)))
        indices_set_3 = np.array(list(itertools.combinations(indices, 3)))    

        neighbor_distances_set_3 = []
        for neighbor in neighbors:
            center_distances = []
            for index_set in indices_set_3:
                dist = (
                    np.sqrt(np.sum((neighbor-points[index_set[0]])**2)),
                    np.sqrt(np.sum((neighbor-points[index_set[1]])**2)),
                    np.sqrt(np.sum((neighbor-points[index_set[2]])**2))
                )
                center_distances.append(np.std(dist))
            neighbor_distances_set_3.append(center_distances)
        neighbor_distances_set_3 = np.array(neighbor_distances_set_3)

        neighbor_center_distances = neighbor_distances_set_3.min(axis=1)
        neighbor_center_indices = neighbor_distances_set_3.argmin(axis=1)
        if verbose:
            print('Minimum Neighbor-Center Distance:',neighbor_center_distances.min())

        dist_std_tolerance = neighbor_center_distances.min()*std_tolerance_multiplier
        if verbose:
            print('Number of Vertices identified:',len(neighbor_center_indices[(neighbor_center_distances <= dist_std_tolerance)]))

        voronoi_point_indices = indices_set_3[neighbor_center_indices[(neighbor_center_distances <= dist_std_tolerance)]]
        voronoi_points = points[voronoi_point_indices]
        voronoi_vertices = neighbors[(neighbor_center_distances <= dist_std_tolerance)]

        vertex_radius = []
        for i in range(len(voronoi_vertices)):
            dist = (
                np.sqrt(np.sum((voronoi_vertices[i]-voronoi_points[i][0])**2)),
                np.sqrt(np.sum((voronoi_vertices[i]-voronoi_points[i][1])**2)),
                np.sqrt(np.sum((voronoi_vertices[i]-voronoi_points[i][2])**2))
            )
            vertex_radius.append((i, np.mean(dist)))

        return voronoi_vertices, voronoi_points, vertex_radius

def get_largest_empty_circle(voronoi_vertices, voronoi_points, vertex_radius):
    LEC_center = voronoi_vertices[max(vertex_radius, key=lambda x: x[1])[0]]
    LEC_radius = max(vertex_radius, key=lambda x: x[1])[1]

    return LEC_center, LEC_radius

def plot_voronoi(centers, dataset, voronoi_vertices, voronoi_points, vertex_radius, LEC_center, LEC_radius):
    points = centers
    neighbors = dataset
    
    plt.scatter(points[:,0], points[:,1])
    plt.scatter(neighbors[:,0],neighbors[:,1])
    plt.scatter(voronoi_vertices[:,0],voronoi_vertices[:,1], marker='x', color='black')

    for i,r in vertex_radius:
        circle = plt.Circle(tuple(voronoi_vertices[i]), r, fill = False, color='y' )
        plt.gca().add_patch(circle)
        pi = np.random.randint(3)
        plt.plot((voronoi_vertices[i][0],voronoi_points[i][pi][0]),(voronoi_vertices[i][1],voronoi_points[i][pi][1]), color='y')

    circle = plt.Circle(tuple(LEC_center), LEC_radius, color='g', alpha=0.2)
    plt.gca().add_patch(circle)

    for i,p in enumerate(voronoi_vertices):
        label = "V{}".format(i)
        plt.annotate(label, # this is the text
                     (p[0],p[1]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center')

    for i,p in enumerate(points):
        label = "C{}".format(i)
        plt.annotate(label, # this is the text
                     (p[0],p[1]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center')

    plt.gca().set_aspect(1)
    plt.show()