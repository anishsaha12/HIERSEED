import numpy as np
import matplotlib.pyplot as plt

def get_other_sub_topic_seed(main_topic_pt, sub_topic_pts): # , threshold):
    dir_vecs = sub_topic_pts-main_topic_pt
    unit_dir_vecs = dir_vecs / np.expand_dims(np.linalg.norm(dir_vecs, axis=-1), axis=1)
    unit_sub_topic_pts = unit_dir_vecs+main_topic_pt

    unit_sub_topic_centroid = np.mean(unit_sub_topic_pts,axis=0)
    other_dir_vec = main_topic_pt-unit_sub_topic_centroid
    other_unit_dir_vec = other_dir_vec / np.linalg.norm(other_dir_vec)
    other_unit_pt = other_unit_dir_vec+main_topic_pt

    # other_radius = threshold/2.3
    other_radius = np.linalg.norm(sub_topic_pts.mean(axis=0)-main_topic_pt)*1
    other_sub_topic_pt = (other_unit_dir_vec*other_radius)+main_topic_pt

    return other_sub_topic_pt,other_radius, (unit_sub_topic_pts,unit_sub_topic_centroid,other_unit_pt)

def plot_other_discovery(main_topic_pt,sub_topic_pts,other_sub_topic_pt,other_radius,unit_sub_topic_pts,\
                         unit_sub_topic_centroid,other_unit_pt,threshold):
    plt.scatter(sub_topic_pts[:,0],sub_topic_pts[:,1])
    plt.scatter(main_topic_pt[0],main_topic_pt[1])
    plt.scatter(unit_sub_topic_pts[:,0],unit_sub_topic_pts[:,1])
    plt.scatter(unit_sub_topic_centroid[0],unit_sub_topic_centroid[1], c='r')
    plt.scatter(other_unit_pt[0],other_unit_pt[1], c='b')
    plt.scatter(other_sub_topic_pt[0],other_sub_topic_pt[1], c='b')

    plt.plot((other_sub_topic_pt[0],unit_sub_topic_centroid[0]),(other_sub_topic_pt[1],unit_sub_topic_centroid[1]), ls=':', c='b')

    circle = plt.Circle(main_topic_pt, 1, fill = False, color='y' )
    plt.gca().add_patch(circle)

    circle = plt.Circle(other_sub_topic_pt, other_radius, fill = False, color='b', ls=':')
    plt.gca().add_patch(circle)

    for i,sub in enumerate(sub_topic_pts):
        plt.plot((main_topic_pt[0],sub[0]),(main_topic_pt[1],sub[1]), ls='--', c='y')

        label = "child{}".format(i)
        plt.annotate(label, # this is the text
                     (sub[0],sub[1]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(8,5), # distance from text to points (x,y)
                     ha='center')

    plt.annotate("parent", # this is the text
                 (main_topic_pt[0],main_topic_pt[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center')
    plt.annotate("centroid", # this is the text
                 (unit_sub_topic_centroid[0],unit_sub_topic_centroid[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(30,5), # distance from text to points (x,y)
                 ha='center',
                 c='r')
    plt.annotate("other", # this is the text
                 (other_sub_topic_pt[0],other_sub_topic_pt[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center',
                 c='b')
    plt.gca().set_aspect(1)
    plt.show()
    
def get_other_topic_seed_no_parent(topic_pts): # , threshold):
    main_topic_pt = topic_pts.mean(axis=0)
    dir_vecs = topic_pts-main_topic_pt
    unit_dir_vecs = dir_vecs / np.expand_dims(np.linalg.norm(dir_vecs, axis=-1), axis=1)
    unit_topic_pts = unit_dir_vecs+main_topic_pt

    unit_sub_topic_centroid = np.mean(unit_topic_pts,axis=0)
    other_dir_vec = main_topic_pt-unit_sub_topic_centroid
    other_unit_dir_vec = other_dir_vec / np.linalg.norm(other_dir_vec)
    other_unit_pt = other_unit_dir_vec+main_topic_pt

    # other_radius = threshold/2.3
    other_radius = np.linalg.norm(main_topic_pt-topic_pts, axis=1).mean()*1
    other_sub_topic_pt = (other_unit_dir_vec*other_radius)+main_topic_pt

    return other_sub_topic_pt,other_radius, (unit_topic_pts,unit_sub_topic_centroid,other_unit_pt)

def plot_other_discovery_no_parent(topic_pts,other_sub_topic_pt,other_radius,unit_topic_pts,\
                         unit_sub_topic_centroid,other_unit_pt,threshold):
    main_topic_pt = topic_pts.mean(axis=0)
    
    plt.scatter(topic_pts[:,0],topic_pts[:,1])
    plt.scatter(main_topic_pt[0],main_topic_pt[1], c='black')
    plt.scatter(unit_topic_pts[:,0],unit_topic_pts[:,1])
    plt.scatter(unit_sub_topic_centroid[0],unit_sub_topic_centroid[1], c='r')
    plt.scatter(other_unit_pt[0],other_unit_pt[1], c='b')
    plt.scatter(other_sub_topic_pt[0],other_sub_topic_pt[1], c='b')

    plt.plot((other_sub_topic_pt[0],unit_sub_topic_centroid[0]),(other_sub_topic_pt[1],unit_sub_topic_centroid[1]), ls=':', c='b')

    circle = plt.Circle(main_topic_pt, 1, fill = False, color='y' )
    plt.gca().add_patch(circle)

    circle = plt.Circle(other_sub_topic_pt, other_radius, fill = False, color='b', ls=':')
    plt.gca().add_patch(circle)

    for i,sub in enumerate(topic_pts):
        plt.plot((main_topic_pt[0],sub[0]),(main_topic_pt[1],sub[1]), ls='--', c='y')

        label = "topic{}".format(i)
        plt.annotate(label, # this is the text
                     (sub[0],sub[1]), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(8,5), # distance from text to points (x,y)
                     ha='center')

    plt.annotate("ctrd", # this is the text
                 (main_topic_pt[0],main_topic_pt[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center',
                 c='black')
    plt.annotate("u_ctrd", # this is the text
                 (unit_sub_topic_centroid[0],unit_sub_topic_centroid[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(20,5), # distance from text to points (x,y)
                 ha='center',
                 c='r')
    plt.annotate("other", # this is the text
                 (other_sub_topic_pt[0],other_sub_topic_pt[1]), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center',
                 c='b')
    plt.gca().set_aspect(1)
    plt.show()