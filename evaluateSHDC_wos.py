import pickle
import json
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import homogeneity_completeness_v_measure
import bcubed
from b3 import calc_b3

def evaluate(dataset_name, topic_seeds, eval_method, topics_points_hierarchical, \
             topics_points_distances_hierarchical, sub_topics_points_distances, \
             eval_full=True, use_test_set=False):
    # duplicate weights to avoid updating them
    topic_seeds = copy.deepcopy(topic_seeds)
    
    # read data indices
    text_embedding_dir = '../data/'+dataset_name
    data_indices_test_file = text_embedding_dir+'/data_indices_test.json'
    data_indices_train_file = text_embedding_dir+'/data_indices_train.json'
    if use_test_set==False:
        with open(data_indices_train_file, "r") as infile:
            data_indices = json.load(infile)
    else:
        with open(data_indices_test_file, "r") as infile:
            data_indices = json.load(infile)
    
    # read annotated set
    base_dir = '../../data/WOS/'
    data_file = base_dir+'Meta-data/Data.csv'
    df = pd.read_csv(data_file).loc[data_indices].rename(
        columns={
            'Y':'level_1',
            'Y1':'level_2',
            'Y2':'level_3'
        }
    )
    df.level_1 = dataset_name
    df.level_2 = df.level_2.astype(str)
    df.level_3 = df.level_3.astype(str)
    
    data_indices = np.array(data_indices)
    
    # Assign to DF section START 
    df['assigned_cluster'] = 'None'
    df['assigned_level_1'] = 'None'
    df['assigned_level_2'] = 'None'
    df['assigned_level_3'] = 'None'
    df['distance_assigned_level_2'] = np.nan
    df['distance_assigned_level_3'] = np.nan
    for topic_hierarchical in topics_points_hierarchical.keys():
        topic_hierarchical_indices = data_indices[topics_points_hierarchical[topic_hierarchical]]
        df.loc[topic_hierarchical_indices, ['assigned_level_1']] = dataset_name
        df.loc[topic_hierarchical_indices, ['assigned_level_2']]  = topic_hierarchical
        df.loc[topic_hierarchical_indices, ['assigned_level_3']]  = 'Other'
        df.loc[topic_hierarchical_indices, ['assigned_cluster']]  = topic_hierarchical
        df.loc[topic_hierarchical_indices, ['distance_assigned_level_2']] = topics_points_distances_hierarchical[topic_hierarchical]
        if topic_hierarchical in sub_topics_points_distances.keys():      # if the topic is actually a sub-topic
            df.loc[topic_hierarchical_indices, ['distance_assigned_level_3']] = sub_topics_points_distances[topic_hierarchical]
            main,sub = topic_hierarchical.split('/')
            df.loc[topic_hierarchical_indices, ['assigned_level_2']]  = main
            df.loc[topic_hierarchical_indices, ['assigned_level_3']]  = sub

    df.loc[(df['level_1']!=dataset_name),'level_1'] = 'None'
    df.loc[(df['level_1']!=dataset_name),'level_2'] = 'None'
    df.loc[(df['level_1']!=dataset_name),'level_3'] = 'None'

    # Assign to DF section END

    if (eval_full==False) and (use_test_set==False):
        eval_indices_file = text_embedding_dir+'/eval_indices.json'
        with open(eval_indices_file, "r") as infile:
            eval_set_indices = json.load(infile)
        df = df.loc[eval_set_indices]
    elif (eval_full==False) and (use_test_set==True):
        eval_indices_file = text_embedding_dir+'/eval_set_test_indices.json'
        with open(eval_indices_file, "r") as infile:
            eval_set_indices = json.load(infile)
        df = df.loc[eval_set_indices]
    
    if eval_method=='hcv':
        homogeneity_l1, completeness_l1, v_measure_l1 = homogeneity_completeness_v_measure(df.level_1, df.assigned_level_1)
        accuracy_l1 = (df.level_1==df.assigned_level_1).sum()/len(df)

        homogeneity_l2, completeness_l2, v_measure_l2 = homogeneity_completeness_v_measure(
            df[df['level_1']==dataset_name].level_2, df[df['level_1']==dataset_name].assigned_level_2
        )
        accuracy_l2 = (df.level_2==df.assigned_level_2).sum()/len(df)

        homogeneity_l3, completeness_l3, v_measure_l3 = homogeneity_completeness_v_measure(
            df[df['level_1']==dataset_name].level_3, df[df['level_1']==dataset_name].assigned_level_3
        )
        accuracy_l3 = (df.level_3==df.assigned_level_3).sum()/len(df)

        metrics = {
            'L1' : {
                'accuracy' : accuracy_l1,
                'homogeneity' : homogeneity_l1,
                'completeness' : completeness_l1,
                'v_measure' : v_measure_l1
            },
            'L2' : {
                'accuracy' : accuracy_l2,
                'homogeneity' : homogeneity_l2,
                'completeness' : completeness_l2,
                'v_measure' : v_measure_l2
            },
            'L3' : {
                'accuracy' : accuracy_l3,
                'homogeneity' : homogeneity_l3,
                'completeness' : completeness_l3,
                'v_measure' : v_measure_l3
            },
            'size' : df[(df.assigned_level_1!='None')].shape[0]
        }

        return metrics
    
    elif eval_method=='bcubed_1':
        precision_l1, recall_l1, fscore_l1 = get_bcubed_metrics(df.assigned_level_1, df.level_1)
        accuracy_l1 = (df.level_1==df.assigned_level_1).sum()/len(df)

        precision_l2, recall_l2, fscore_l2 = get_bcubed_metrics(
            df[df['level_1']==dataset_name].assigned_level_2, df[df['level_1']==dataset_name].level_2
        )
        accuracy_l2 = (df.level_2==df.assigned_level_2).sum()/len(df)

        precision_l3, recall_l3, fscore_l3 = get_bcubed_metrics(
            df[df['level_1']==dataset_name].assigned_level_3, df[df['level_1']==dataset_name].level_3
        )
        accuracy_l3 = (df.level_3==df.assigned_level_3).sum()/len(df)

        metrics = {
            'L1' : {
                'accuracy' : accuracy_l1,
                'precision' : precision_l1,
                'recall' : recall_l1,
                'fscore' : fscore_l1
            },
            'L2' : {
                'accuracy' : accuracy_l2,
                'precision' : precision_l2,
                'recall' : recall_l2,
                'fscore' : fscore_l2
            },
            'L3' : {
                'accuracy' : accuracy_l3,
                'precision' : precision_l3,
                'recall' : recall_l3,
                'fscore' : fscore_l3
            },
            'size' : df[(df.assigned_level_1!='None')].shape[0]
        }
        
        return metrics
    
    elif eval_method=='bcubed_2':
        precision_l1, recall_l1, fscore_l1 = get_bcubed_metrics(df.assigned_level_1, df.level_1)
        accuracy_l1 = (df.level_1==df.assigned_level_1).sum()/len(df)

        precision_l2, recall_l2, fscore_l2 = get_bcubed_metrics(
            df.assigned_level_2, df.level_2
        )
        accuracy_l2 = (df.level_2==df.assigned_level_2).sum()/len(df)

        precision_l3, recall_l3, fscore_l3 = get_bcubed_metrics(
            df.assigned_level_3, df.level_3
        )
        accuracy_l3 = (df.level_3==df.assigned_level_3).sum()/len(df)

        metrics = {
            'L1' : {
                'accuracy' : accuracy_l1,
                'precision' : precision_l1,
                'recall' : recall_l1,
                'fscore' : fscore_l1
            },
            'L2' : {
                'accuracy' : accuracy_l2,
                'precision' : precision_l2,
                'recall' : recall_l2,
                'fscore' : fscore_l2
            },
            'L3' : {
                'accuracy' : accuracy_l3,
                'precision' : precision_l3,
                'recall' : recall_l3,
                'fscore' : fscore_l3
            },
            'size' : df[(df.assigned_level_1!='None')].shape[0]
        }

        return metrics
    
    elif eval_method=='both':
        homogeneity_l1, completeness_l1, v_measure_l1 = homogeneity_completeness_v_measure(df.level_1, df.assigned_level_1)

        homogeneity_l2, completeness_l2, v_measure_l2 = homogeneity_completeness_v_measure(
            df[df.level_1==df.assigned_level_1].level_2, df[df.level_1==df.assigned_level_1].assigned_level_2
        )

        homogeneity_l3, completeness_l3, v_measure_l3 = homogeneity_completeness_v_measure(
            df[df.level_1==df.assigned_level_1].level_3, df[df.level_1==df.assigned_level_1].assigned_level_3
        )
        
        # precision_l1, recall_l1, fscore_l1 = get_bcubed_metrics(df.assigned_level_1, df.level_1)
        fscore_l1,precision_l1,recall_l1 = calc_b3(df.assigned_level_1, df.level_1)
        accuracy_l1 = (df.level_1==df.assigned_level_1).sum()/len(df[df.level_1==df.assigned_level_1])

        fscore_l2,precision_l2,recall_l2 = calc_b3(
            df[df.level_1==df.assigned_level_1].assigned_level_2, df[df.level_1==df.assigned_level_1].level_2
        )
        accuracy_l2 = (df.level_2==df.assigned_level_2).sum()/len(df[df.level_1==df.assigned_level_1])

        fscore_l3,precision_l3,recall_l3 = calc_b3(
            df[df.level_1==df.assigned_level_1].assigned_level_3, df[df.level_1==df.assigned_level_1].level_3
        )
        accuracy_l3 = (df.level_3==df.assigned_level_3).sum()/len(df[df.level_1==df.assigned_level_1])

        metrics = {
            'L1' : {
                'accuracy' : accuracy_l1,
                'homogeneity' : homogeneity_l1,
                'completeness' : completeness_l1,
                'v_measure' : v_measure_l1,
                'precision' : precision_l1,
                'recall' : recall_l1,
                'fscore' : fscore_l1
            },
            'L2' : {
                'accuracy' : accuracy_l2,
                'homogeneity' : homogeneity_l2,
                'completeness' : completeness_l2,
                'v_measure' : v_measure_l2,
                'precision' : precision_l2,
                'recall' : recall_l2,
                'fscore' : fscore_l2
            },
            'L3' : {
                'accuracy' : accuracy_l3,
                'homogeneity' : homogeneity_l3,
                'completeness' : completeness_l3,
                'v_measure' : v_measure_l3,
                'precision' : precision_l3,
                'recall' : recall_l3,
                'fscore' : fscore_l3
            },
            'size' : df[(df.assigned_level_1!='None')].shape[0]
        }
        
        return metrics
    
    else:
        return None

def get_b_dict(column):
    return dict(
        zip(
            [item for (item,cluster) in column.to_dict().items()],
            [set([cluster]) for (item,cluster) in column.to_dict().items()]
        )
    )

def get_bcubed_metrics(clustered, labelled):
    cdict = get_b_dict(clustered)
    ldict = get_b_dict(labelled)
    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)
    return (precision, recall, fscore)

def print_metrics(metrics):
    print('Level 1 -')
    print('|- accuracy:',round(metrics['L1']['accuracy'],4))
    print('|- homogeneity:',round(metrics['L1']['homogeneity'],4))
    print('|- completeness:',round(metrics['L1']['completeness'],4))
    print('|- v_measure:',round(metrics['L1']['v_measure'],4))
    print('    Level 2 -')
    print('    |- accuracy:',round(metrics['L2']['accuracy'],4))
    print('    |- homogeneity:',round(metrics['L2']['homogeneity'],4))
    print('    |- completeness:',round(metrics['L2']['completeness'],4))
    print('    |- v_measure:',round(metrics['L2']['v_measure'],4))
    print('        Level 3 -')
    print('        |- accuracy:',round(metrics['L3']['accuracy'],4))
    print('        |- homogeneity:',round(metrics['L3']['homogeneity'],4))
    print('        |- completeness:',round(metrics['L3']['completeness'],4))
    print('        |- v_measure:',round(metrics['L3']['v_measure'],4))

def print_metrics_bcubed(metrics):
    print('Level 1 -')
    print('|- accuracy:',round(metrics['L1']['accuracy'],4))
    print('|- precision:',round(metrics['L1']['precision'],4))
    print('|- recall:',round(metrics['L1']['recall'],4))
    print('|- fscore:',round(metrics['L1']['fscore'],4))
    print('    Level 2 -')
    print('    |- accuracy:',round(metrics['L2']['accuracy'],4))
    print('    |- precision:',round(metrics['L2']['precision'],4))
    print('    |- recall:',round(metrics['L2']['recall'],4))
    print('    |- fscore:',round(metrics['L2']['fscore'],4))
    print('        Level 3 -')
    print('        |- accuracy:',round(metrics['L3']['accuracy'],4))
    print('        |- precision:',round(metrics['L3']['precision'],4))
    print('        |- recall:',round(metrics['L3']['recall'],4))
    print('        |- fscore:',round(metrics['L3']['fscore'],4))