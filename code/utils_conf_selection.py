from utils import json2dict
from copy import deepcopy
import time
import random

def get_ner_idx_dict(label_list, data, label_key=None):
    '''
    Mapping: label to sample indexes (dict). 
             Format: {label1: [ind1, ind2, ...], label2: [ind3, ind4, ...], ...}
    '''
    # Initialize mapping
    ner_dict = {}
    for lb in label_list:
        ner_dict[lb] = set()

    # Collect mapping from dataset
    for i_item, item in enumerate(data):
        label = item[label_key]
        if isinstance(label, str):
            label = eval(label)
        if isinstance(label, list):
            label = json2dict(label)
        for _, v in label.items():
            
            # remove OOD type
            if v not in ner_dict:
                continue

            ner_dict[v].add(i_item)
    for lb in ner_dict:
        ner_dict[lb] = list(ner_dict[lb])
    print("\nStatistics of label2inds mapping:")
    print([(k, len(ner_dict[k])) for k in ner_dict])
    print(f"# appeared labels = {len(ner_dict)}")
    return ner_dict

def update_ner_idx_dict(ner_idx_dict, removing_idx=None):
    for lb in ner_idx_dict:
        if removing_idx in ner_idx_dict[lb]:
            ner_idx_dict[lb].remove(removing_idx)
    
def update_label_count(count_label_shots, removing_label=None, adding_label=None):
    if adding_label and removing_label:
        raise ValueError(f"Choose only one between removing and adding label.")
    if adding_label:
        for k,v in adding_label.items():

            # remove OOD type
            if v not in count_label_shots:
                continue

            count_label_shots[v] += 1
    elif removing_label:
        for k,v in removing_label.items():

            # remove OOD type
            if v not in count_label_shots:
                continue

            count_label_shots[v] -= 1
    else:
        raise ValueError(f"Choose at least one between removing and adding label.")

def select_by_sample_threshold(args, pred_data):
    # TODO: check
    selected_data = []
    for _, item in enumerate(pred_data):
        sc_score = item["consistency_score"]
        if isinstance(sc_score, str):
            sc_score = eval(sc_score)
        sc_score_avg = sc_score["avg"]
        if sc_score_avg >= args.sample_threshold:
            selected_data.append(item)
    return selected_data

def select_by_random(args, pred_data):
    random.seed(args.random_seed)

    indices = list(range(len(pred_data)))
    rd_sampled_indices = random.sample(indices, args.confident_sample_size)

    selected_data = []
    for idx in rd_sampled_indices:
        selected_data.append(pred_data[idx])
    return selected_data
    
def select_by_entity_threshold(args, pred_data):
    selected_data = []
    entity_threshold = args.entity_threshold
    for _, item in enumerate(pred_data):
        sc_score_all_ans = item["consistency_score_SC_all_ans"]
        if isinstance(sc_score_all_ans, str):
            sc_score_all_ans = eval(sc_score_all_ans)
        sc_score_ent_pairs = sc_score_all_ans["entities"]

        pred = {}
        for k,v in sc_score_ent_pairs.items():
            if v < entity_threshold:
                continue
            tmp_mention, tmp_type = k[0], k[1]
            # solve type conflicts: choose the one with higher score
            if tmp_mention in pred:
                tmp_previous_type = pred[tmp_mention]
                if sc_score_ent_pairs[(tmp_mention, tmp_type)] < sc_score_ent_pairs[(tmp_mention, tmp_previous_type)]:
                    continue
            pred[tmp_mention] = tmp_type

        sc_score_pred = {}
        for tmp_mention, tmp_type in pred.items():
            sc_score_pred[(tmp_mention, tmp_type)] = sc_score_ent_pairs[(tmp_mention, tmp_type)]
        
        sc_score_pred_avg = sum(list(sc_score_pred.values())) / len(sc_score_pred) if sc_score_pred else 0

        item_new = deepcopy(item)
        item_new["prediction"] = pred
        item_new["consistency_score"] = {"entities": sc_score_pred, "avg": sc_score_pred_avg}
        
        selected_data.append(item_new)

    return selected_data

def preserve_all_entities(args, pred_data):
    '''preserve all entities, solve type conflicts'''
    selected_data = []
    for _, item in enumerate(pred_data):
        sc_score_all_ans = item["consistency_score_SC_all_ans"]
        if isinstance(sc_score_all_ans, str):
            sc_score_all_ans = eval(sc_score_all_ans)
        sc_score_ent_pairs = sc_score_all_ans["entities"]

        pred = {}
        for k,_ in sc_score_ent_pairs.items():
            tmp_mention, tmp_type = k[0], k[1]
            # solve type conflicts: choose the one with higher score
            if tmp_mention in pred:
                tmp_previous_type = pred[tmp_mention]
                if sc_score_ent_pairs[(tmp_mention, tmp_type)] < sc_score_ent_pairs[(tmp_mention, tmp_previous_type)]:
                    continue
            pred[tmp_mention] = tmp_type

        sc_score_pred = {}
        for tmp_mention, tmp_type in pred.items():
            sc_score_pred[(tmp_mention, tmp_type)] = sc_score_ent_pairs[(tmp_mention, tmp_type)]
        
        sc_score_pred_avg = sum(list(sc_score_pred.values())) / len(sc_score_pred) if sc_score_pred else 0

        item_new = deepcopy(item)
        item_new["prediction"] = pred
        item_new["consistency_score"] = {"entities": sc_score_pred, "avg": sc_score_pred_avg}
        
        selected_data.append(item_new)

    return selected_data

def TSMV_with_sc_scores(args, pred_data):
    '''
    Using SC scores of all SC answers for TSMV (two-stage majority voting).
        e.g., "consistency_score_SC_all_ans": "{'entities': {('Rolf Sorensen', 'Person'): 5, ('Denmark', 'Location'): 5, ('Rabobank', 'Organization'): 5}, 'avg': 5.0}"
    '''
    n_majority_votes = args.query_times // 2
    if args.query_times % 2 == 1:
        n_majority_votes += 1

    pred_data_TSMV = []
    for _, item in enumerate(pred_data):
        sc_score_all_ans = item["consistency_score_SC_all_ans"]
        if isinstance(sc_score_all_ans, str):
            sc_score_all_ans = eval(sc_score_all_ans)
        sc_score_ent_pairs = sc_score_all_ans["entities"]

        mention2count = {}
        for k,v in sc_score_ent_pairs.items():
            tmp_mention, _ = k[0], k[1]
            if tmp_mention not in mention2count:
                mention2count[tmp_mention] = 0
            mention2count[tmp_mention] += v
        
        majority_mentions = []
        for mention, count in mention2count.items():
            if count >= n_majority_votes:
                majority_mentions.append(mention)
        
        pred = {}
        for k,v in sc_score_ent_pairs.items():
            tmp_mention, tmp_type = k[0], k[1]
            # only keep majority voted mentions
            if tmp_mention not in majority_mentions:
                continue
            # solve type conflicts: choose the one with higher score
            if tmp_mention in pred:
                tmp_previous_type = pred[tmp_mention]
                if sc_score_ent_pairs[(tmp_mention, tmp_type)] < sc_score_ent_pairs[(tmp_mention, tmp_previous_type)]:
                    continue
            pred[tmp_mention] = tmp_type

            sc_score_pred = {}
            for tmp_mention, tmp_type in pred.items():
                sc_score_pred[(tmp_mention, tmp_type)] = sc_score_ent_pairs[(tmp_mention, tmp_type)]
            
            sc_score_pred_avg = sum(list(sc_score_pred.values())) / len(sc_score_pred) if sc_score_pred else 0

            item_new = deepcopy(item)
            item_new["prediction"] = pred
            item_new["consistency_score"] = {"entities": sc_score_pred, "avg": sc_score_pred_avg}
            
            pred_data_TSMV.append(item_new)
    
    return pred_data_TSMV
        

def MV_with_sc_scores(args, pred_data):
    '''
    Using SC scores of all SC answers for MV (majority voting).
        e.g., "consistency_score_SC_all_ans": "{'entities': {('Rolf Sorensen', 'Person'): 5, ('Denmark', 'Location'): 5, ('Rabobank', 'Organization'): 5}, 'avg': 5.0}"
    '''
    n_majority_votes = args.query_times // 2
    if args.query_times % 2 == 1:
        n_majority_votes += 1

    pred_data_MV = []
    for _, item in enumerate(pred_data):
        sc_score_all_ans = item["consistency_score_SC_all_ans"]
        if isinstance(sc_score_all_ans, str):
            sc_score_all_ans = eval(sc_score_all_ans)
        sc_score_ent_pairs = sc_score_all_ans["entities"]

        pred = {}
        for k,v in sc_score_ent_pairs.items():
            if v < n_majority_votes:
                continue
            tmp_mention, tmp_type = k[0], k[1]
            # solve type conflicts: choose the one with higher score
            if tmp_mention in pred:
                tmp_previous_type = pred[tmp_mention]
                if sc_score_ent_pairs[(tmp_mention, tmp_type)] < sc_score_ent_pairs[(tmp_mention, tmp_previous_type)]:
                    continue
            pred[tmp_mention] = tmp_type

            sc_score_pred = {}
            for tmp_mention, tmp_type in pred.items():
                sc_score_pred[(tmp_mention, tmp_type)] = sc_score_ent_pairs[(tmp_mention, tmp_type)]
            
            sc_score_pred_avg = sum(list(sc_score_pred.values())) / len(sc_score_pred) if sc_score_pred else 0

            item_new = deepcopy(item)
            item_new["prediction"] = pred
            item_new["consistency_score"] = {"entities": sc_score_pred, "avg": sc_score_pred_avg}
            
            pred_data_MV.append(item_new)
    
    return pred_data_MV