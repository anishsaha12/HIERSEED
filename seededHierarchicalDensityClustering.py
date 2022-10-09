import copy
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np
from voronoi_lec import voronoi_cluster, get_largest_empty_circle
from other_seed import get_other_sub_topic_seed, get_other_topic_seed_no_parent

class SeededHierarchicalDensityClustering:
    def __init__(self, topic_seeds, init_neighbors_perc=0.1, main_topic_update_method=1, main_topic_weight=3, \
                 create_other_main=False, create_other_sub=False, verbose=False):
        self.topic_seeds = topic_seeds
        self.init_neighbors_perc = init_neighbors_perc
        self.main_topics = [topic for topic in topic_seeds.keys() if (topic.find('/')==-1 and topic!='None')]
        self.main_topic_seeds = np.vstack([topic_seeds[topic] for topic in self.main_topics])
        self.main_topic_update_method = main_topic_update_method
        self.main_topic_weight = main_topic_weight
        self.verbose = verbose
        self.create_other_main = create_other_main
        self.create_other_sub = create_other_sub
        
        if create_other_main==True:
            ## create Other seed at main_topics level
            other_main_topic,other_radius, _ = get_other_topic_seed_no_parent(self.main_topic_seeds)
            topic_seeds['Other'] = other_main_topic
            self.topic_seeds['Other'] = other_main_topic
            self.main_topics = [topic for topic in self.topic_seeds.keys() if (topic.find('/')==-1 and topic!='None')]
            self.main_topic_seeds = np.vstack([self.topic_seeds[topic] for topic in self.main_topics])
            if self.verbose:
                print('Created Other main-topic...')
                
        if create_other_sub==True:
            ## create Other seed at sub_topics level for each main_topic
            for main_topic in self.main_topics:
                main_topic_seed = self.topic_seeds[main_topic]
                sub_topics = [topic for topic in self.topic_seeds.keys() if (topic.startswith(main_topic+'/'))]
                if len(sub_topics)!=0 :   # sub topics exist
                    sub_topic_seeds = np.array([self.topic_seeds[sub_topic] for sub_topic in sub_topics])
                    other_sub_topic, other_radius, _ = get_other_sub_topic_seed(main_topic_seed, sub_topic_seeds)
                    topic_seeds[main_topic+'/Other'] = other_sub_topic
                    self.topic_seeds[main_topic+'/Other'] = other_sub_topic
                    if self.verbose:
                        print('Created Other sub-topic for',main_topic,'...')
    
    def _create_dist_matrix(self, X, topic_seeds):
        init_n_neighbors = int(X.shape[0]*self.init_neighbors_perc)
        neigh = NearestNeighbors(n_neighbors=init_n_neighbors)
        neigh = neigh.fit(X)
        main_topic_seeds = np.vstack([topic_seeds[topic] for topic in self.main_topics])
        topics_distances, topics_neighbors = neigh.kneighbors(main_topic_seeds, return_distance=True)
        self.topics_distances = topics_distances
        self.topics_neighbors = topics_neighbors
        return
    
    def _get_topic_distance_threshold(self, topic_seeds, topic_distance_threshold={}):
        for i,main_topic in enumerate(self.main_topics):
            sub_topics = [topic for topic in topic_seeds.keys() if (topic.startswith(main_topic+'/'))]
            if len(sub_topics)!=0:     # if sub-topics exist
                dist = [np.sqrt(np.sum((topic_seeds[main_topic]-topic_seeds[sub_topic])**2)) for sub_topic in sub_topics]
                threshold = max(dist)*2.3
                if threshold <= self.topics_distances[i].min():      # if determined threshold is still less than the nearest text
                    if self.topics_distances[i].min() <= 2*threshold:    # if nearest text is not too far 
                        threshold = 1.1*self.topics_distances[i].min()
                    else:
                        threshold = 0
            else:
                dist = [
                    np.sqrt(
                        np.sum((topic_seeds[main_topic]-topic_seeds[topic])**2)
                    ) for topic in self.main_topics if topic!=main_topic
                ]
                threshold = min(dist)*1.3
                if threshold <= self.topics_distances[i].min():      # if determined threshold is still less than the nearest text
                    if self.topics_distances[i].min() <= 2*threshold:     # if nearest text is not too far 
                        threshold = 1.1*self.topics_distances[i].min()
                    else:
                        threshold = 0
            topic_distance_threshold[main_topic] = threshold
        return topic_distance_threshold
    
    def _get_L1_topic_assignments(self, topic_distance_threshold):
        topics_candidate_points = {}
        topics_candidate_distances = {}
        for main_topic, topic_distances, topic_neighbors, topic_threshold in zip(
            self.main_topics, self.topics_distances, self.topics_neighbors, topic_distance_threshold.values()
        ):
            topics_candidate_points[main_topic] = topic_neighbors[topic_distances <= topic_threshold]
            topics_candidate_distances[main_topic] = topic_distances[topic_distances <= topic_threshold]

        if self.verbose:
            print(np.unique(np.concatenate(list(topics_candidate_points.values()))).shape[0], 'texts have been assigned to topics')
        points, counts = np.unique(np.concatenate(list(topics_candidate_points.values())), return_counts=True)

        # points that are in more than one topic cluster, assign to closest topic
        multi_topic_pts = points[counts>1]
        if self.verbose:
            print(multi_topic_pts.shape[0],'texts are in more than one topic')

        topics_points = {}
        topics_points_distances = {}
        for main_topic in self.main_topics:
            indices = ~(np.in1d(topics_candidate_points[main_topic], multi_topic_pts))
            topics_points[main_topic] = topics_candidate_points[main_topic][indices]
            topics_points_distances[main_topic] = topics_candidate_distances[main_topic][indices]

        for point in multi_topic_pts:
            dists = []
            for main_topic in self.main_topics:
                try:
                    dists.append(topics_candidate_distances[main_topic][point == topics_candidate_points[main_topic]][0])
                except:
                    dists.append(float('inf'))
            closest_topic = self.main_topics[np.argmin(dists)]
            closest_topic_distance = np.min(dists)
            topics_points[closest_topic] = np.append(topics_points[closest_topic], point)
            topics_points_distances[closest_topic] = np.append(topics_points_distances[closest_topic], closest_topic_distance)

        if self.verbose:
            print(np.concatenate(list(topics_points.values())).shape[0], 'texts have been assigned to unique topics')
        self.topics_points = topics_points
        self.topics_points_distances = topics_points_distances
        return
    
    def _get_L2_topic_assignments(self, X, topics_points, topics_points_distances):
        # Perform k-Means within each main topic, using the sub topic embeddings as centers
        topic_seeds_updated = {}
        topics_points_hierarchical = {}
        topics_points_distances_hierarchical = {}
        sub_topics_points_distances = {}

        for main_topic in self.main_topics:
            topic_seeds_updated[main_topic] = self.topic_seeds[main_topic]
            sub_topics = [topic for topic in self.topic_seeds.keys() if (topic.startswith(main_topic+'/'))]
            if len(sub_topics)>0:
                if self.verbose:
                    print('Clustering:',', '.join(sub_topics))
                sub_topic_seeds = [self.topic_seeds[sub_topic] for sub_topic in sub_topics]
                sub_topic_seeds = np.vstack(sub_topic_seeds)

                n_clusters = len(sub_topic_seeds)
                mbkm = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    init=sub_topic_seeds,
                    random_state=20,
                    n_init=1
                )
                clusters = mbkm.fit_predict(X[topics_points[main_topic]])
                clusters = np.vectorize(lambda x: sub_topics[x])(clusters)
                for i,sub_topic in enumerate(sub_topics):
                    clust_pts_index = (clusters==sub_topic)
                    topic_seeds_updated[sub_topic] = mbkm.cluster_centers_[i]
                    topics_points_hierarchical[sub_topic] = topics_points[main_topic][clust_pts_index]
                    topics_points_distances_hierarchical[sub_topic] = topics_points_distances[main_topic][clust_pts_index]
                    sub_topics_points_distances[sub_topic] = np.sqrt(
                        np.sum(
                            ( X[topics_points[main_topic][clust_pts_index]] - mbkm.cluster_centers_[i] )**2, 
                            axis=1
                        )
                    )

            else:
                topics_points_hierarchical[main_topic] = topics_points[main_topic]
                topics_points_distances_hierarchical[main_topic] = topics_points_distances[main_topic]
        self.topics_points_hierarchical = topics_points_hierarchical
        self.topics_points_distances_hierarchical = topics_points_distances_hierarchical
        self.sub_topics_points_distances = sub_topics_points_distances
        self._update_topic_seeds(topic_seeds_updated, X, self.main_topic_update_method, self.main_topic_weight)
        return
    
    def _update_topic_seeds(self, topic_seeds, X, main_topic_update_method=1, main_topic_weight=3):
        topic_seeds = copy.deepcopy(topic_seeds)
        for main_topic in self.main_topics:
            try:
                sub_topics = [topic for topic in self.topic_seeds.keys() if (topic.startswith(main_topic+'/'))]
                if self.verbose:
                    print('Updating Main Topic',main_topic,':',', '.join(sub_topics))
                # Update topic embedding by taking mean of itself with all sub-topic embeddings
                if main_topic_update_method==1:
                    topic_seeds[main_topic] = np.mean(
                        [
                          topic_seeds[sub_topic] for sub_topic in sub_topics
                        ] + [topic_seeds[main_topic]]*main_topic_weight
                        , axis=0
                    )
                elif main_topic_update_method==2:
                    if len(sub_topics)!=0:
                        topic_seeds[main_topic] = np.mean(
                            [
                                np.mean(
                                    [
                                      topic_seeds[sub_topic] for sub_topic in sub_topics
                                    ]
                                    , axis=0
                                )
                            ]
                            + [topic_seeds[main_topic]]*main_topic_weight
                            , axis=0
                        )
                elif main_topic_update_method==3:
                    if len(sub_topics)!=0:
                        topic_seeds[main_topic] = np.mean(
                            [
                                np.mean(
                                    [
                                      topic_seeds[sub_topic] for sub_topic in sub_topics
                                    ]
                                    , axis=0
                                )
                            ]
                            + [topic_seeds[main_topic]]*main_topic_weight*len(sub_topics)
                            , axis=0
                        )
                elif main_topic_update_method==4:  # Voronoi LEC method to update main topic center embedding
                    if len(sub_topics)!=0:
                        sub_topic_centers = np.array([
                          topic_seeds[sub_topic] for sub_topic in sub_topics
                        ])
                        neighbors = X[self.topics_points[main_topic]]
                        voronoi_vertices, voronoi_points, vertex_radius = voronoi_cluster(
                            sub_topic_centers, neighbors, std_tolerance_multiplier=10, verbose=self.verbose
                        )
                        LEC_center, LEC_radius = get_largest_empty_circle(voronoi_vertices, voronoi_points, vertex_radius)
                        topic_seeds[main_topic] = np.mean(
                            [LEC_center]
                            + [topic_seeds[main_topic]]*main_topic_weight
                            + [np.mean(sub_topic_centers, axis=0)]*1
                            , axis=0
                        )
                else:
                    if self.verbose:
                        print('Main Topic',main_topic,'not updated')
            except:
                if self.verbose:
                    print('Exception: Main Topic',main_topic,'not updated')
                pass
        self.topic_seeds_updated = topic_seeds
        self._create_dist_matrix(X, self.topic_seeds_updated) # temporarily needed to compute new thresholds
        topic_distance_threshold_updated = self._get_topic_distance_threshold(self.topic_seeds_updated, {})
        self.topic_distance_threshold_updated = topic_distance_threshold_updated
        self._create_dist_matrix(X, self.topic_seeds) # get back original as was changed during update
        return
    
    def fit_L1(self, X):
        self._create_dist_matrix(X, self.topic_seeds)
        topic_distance_threshold = self._get_topic_distance_threshold(self.topic_seeds, {})
        self.topic_distance_threshold = topic_distance_threshold
        self._get_L1_topic_assignments(self.topic_distance_threshold)
        return
    
    def fit_L2(self, X):
        try:
            self._get_L2_topic_assignments(X, self.topics_points, self.topics_points_distances)
        except:
            if self.verbose:
                print('computing L1 clusters first...')
            self.fit_L1(X)
            self._get_L2_topic_assignments(X, self.topics_points, self.topics_points_distances)