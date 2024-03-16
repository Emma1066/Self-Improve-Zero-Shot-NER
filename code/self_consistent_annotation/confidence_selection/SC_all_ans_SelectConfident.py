import json
import time
import logging, logging.config
import sys
import os
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import argparse

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) ))

from utils import save_data, load_data
from utils_conf_selection import select_by_sample_threshold, select_by_entity_threshold, preserve_all_entities, TSMV_with_sc_scores, MV_with_sc_scores, select_by_random
from const import model_list

def entity_level_selection(args, pred_data):
    if args.entity_level_selection is None:
        raise ValueError(f"entity_level_selection cannot be None.")
    
    if args.entity_level_selection == "th_ent":
        confident_pred_data = select_by_entity_threshold(args, pred_data)
    elif args.entity_level_selection == "all":
        confident_pred_data = preserve_all_entities(args, pred_data)
    elif args.entity_level_selection == "TSMV":
        if args.consistency_selection == "two_stage_majority_voting":
            return pred_data
        else:
            confident_pred_data = TSMV_with_sc_scores(args, pred_data)
    elif args.entity_level_selection == "MV":
        if args.consistency_selection == "majority_voting":
            return pred_data
        else:
            confident_pred_data = MV_with_sc_scores(args, pred_data)
    else:
        raise ValueError(f"Unrecognized entity_level_selection = {args.entity_level_selection}")
    
    return confident_pred_data

def sample_level_selection(args, pred_data):
    if args.sample_level_selection is None:
        return pred_data
    
    confident_pred_data = None
    if args.sample_level_selection == ["ClEn"]:
        confident_pred_data = select_by_ClEn(args, pred_data)
    elif args.sample_level_selection == ["th_spl"]:
        confident_pred_data = select_by_sample_threshold(args, pred_data)
    elif args.sample_level_selection == ["randCla"]:
        confident_pred_data = random_sample_few_shot_minimum_include(args, pred_data)
    elif args.sample_level_selection == ["KmeansClust"]:
        confident_pred_data = select_by_kmeans_clustering(args, pred_data)
    elif args.sample_level_selection == ["mpKmeansClust"]:
        confident_pred_data = select_by_max_per_kmeans_cluster(args, pred_data)
    elif args.sample_level_selection == ["th_spl", "randCla"]:
        confident_pred_data = select_by_sample_threshold(args, pred_data)
        confident_pred_data = random_sample_few_shot_minimum_include(args, confident_pred_data)
    elif args.sample_level_selection == ["th_spl", "KmeansClust"]:
        confident_pred_data = select_by_sample_threshold(args, pred_data)
        confident_pred_data = select_by_kmeans_clustering(args, confident_pred_data)
    elif args.sample_level_selection == ["th_spl", "ClEn"]:
        confident_pred_data = select_by_sample_threshold(args, pred_data)
        confident_pred_data = select_by_ClEn(args, confident_pred_data)
    elif args.sample_level_selection == ["th_spl", "rand"]:
        confident_pred_data = select_by_sample_threshold(args, pred_data)
        confident_pred_data = select_by_random(args, confident_pred_data)
    else:
        print(type(args.sample_level_selection))
        raise ValueError(f"Unrecognized sample_level_selection = {args.sample_level_selection}")

    return confident_pred_data


def selection_pipeline_entity_first(args, pred_data):
    conf_data = entity_level_selection(args, pred_data)
    conf_data = sample_level_selection(args, conf_data)
    return conf_data

def selection_pipeline_sample_first(args, pred_data):
    conf_data = sample_level_selection(args, pred_data)
    conf_data = entity_level_selection(args, conf_data)
    return conf_data

def select_confident_annotations(args, pred_data):
    if args.first_level == "entity":
        conf_data = selection_pipeline_entity_first(args, pred_data)
    elif args.first_level == "sample":
        conf_data = selection_pipeline_sample_first(args, pred_data)
    else:
        raise ValueError(f"Unrecognized first_level = {args.first_level}")

    return conf_data

def main(args):
    pred_data = load_data(args.pred_path)

    confident_pred_data = select_confident_annotations(args, pred_data)

    # convert to str
    converting_keys = ["consistency_score", "prediction", "label"]
    for item in confident_pred_data:
        for k in converting_keys:
            if k in item:
                item[k] = str(item[k])
    
    # Refine confident data saving path
        # - randCla, needs len(confident_pred_data)
    # args = get_paths(args, conf_data=confident_pred_data)

    save_data(args.confident_pred_path, confident_pred_data)
    print(f"confident samples saved to: {args.confident_pred_path}")
    print(f"confident sample number = {len(confident_pred_data)}")

def get_conf_selection_tag(args):
    # Tag for sample-level selection
    if args.sample_level_selection:
        if args.sample_level_selection == ["rand"]:
            tag_sample_level = f"{args.sample_level_selection[0]}_{args.confident_sample_size}_{args.random_seed}"
        elif args.sample_level_selection == ["th_spl"]:
            tag_sample_level = f"{args.sample_level_selection[0]}_{args.sample_threshold}"
        else:
            raise ValueError(f"Unrecognized sample_level_selection={args.sample_level_selection}")

    # Tag for entity-level selection
    if args.entity_level_selection:
        if args.entity_level_selection in ["TSMV", "MV", "all"]:
            tag_entity_level = args.entity_level_selection
        elif args.entity_level_selection == "th_ent":
            tag_entity_level = f"{args.entity_level_selection}_{args.entity_threshold}"
        else:
            raise ValueError(f"Unrecognized sample_level_selection={args.entity_level_selection}")

    # Combine two level tags
    # raise error if two level selection methods are all None
    if args.sample_level_selection is None and args.entity_level_selection is None:
        raise ValueError(f"Both entity-level and sample-level selection are None!")
    elif args.sample_level_selection is None:
        tag_conf_selection = tag_entity_level
    elif args.entity_level_selection is None:
        tag_conf_selection = tag_sample_level
    else:
        if args.first_level == "entity":
            tag_conf_selection = f"{tag_entity_level}_{tag_sample_level}"
        elif args.first_level == "sample":
            tag_conf_selection = f"{tag_sample_level}_{tag_entity_level}"
        else:
            raise ValueError(f"Unrecognized first_level={args.first_level}")
    
    return tag_conf_selection
    

def get_paths(args, conf_data=None):
    dataname = args.dataname
    datamode = args.datamode

    # label path
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # embedding path
    args.train_GPTEmb_path = f"data/{dataname}/{datamode}_GPTEmb.npy"    
    args.train_SBERTEmb_path = f"data/{dataname}/{datamode}_SBERTEmb.npy"
    
    # prompt path
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder = f"fs_{folder}"
    if args.few_shot_setting in ["fixed", "pool"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder = f"{folder}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder = f"{folder}_tool"

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = time.strftime("%m%d%H%M")
    if args.start_time:
        start_time = args.start_time

    # tag for confident sample selection method
    conf_sel_tag = get_conf_selection_tag(args)

    if args.confidence_scoring_method=="SC":
        pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
        confident_pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{conf_sel_tag}_response.json"
    elif args.confidence_scoring_method=="SV":
        # pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_sv_response.json"
        # confident_pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_sv_{conf_sel_tag}_response.json" 
        raise NotImplementedError(f"Not implemented for self-verification")
    else:
        raise ValueError(f"Unrecognized confidence_scoring_method: {args.confidence_scoring_method}")

    if args.consistency: # add consistency tag
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder_pred = f"{folder}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"

    # Add tags: self-consistent-annotate、self-annotation/self-supervision、demo_datamode (eg., train_shuflle_42)
    sca_folder = "self_consistent_annotate"
    sa_sp_folder = "self_annotation"
    datamode_folder = args.datamode
    model_folder = model_list[args.model]["abbr"]

    pred_dir = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder_pred}"
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    args.pred_path = os.path.join(pred_dir, pred_filename)
    args.confident_pred_path = os.path.join(pred_dir, confident_pred_filename)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=int)
    parser.add_argument("--datamode", default="train", type=str)
    parser.add_argument("--demo_datamode", default=None, type=str)

    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=None, type=int)
    parser.add_argument("--demo_select_method", default=None) # "random", "GPTEmbClusterKmeans"
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)
    
    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    # Source of confidence score：Self-Consistency, Self-verification
    parser.add_argument("--confidence_scoring_method", default="SC", type=str, choices=["SC"])

    # Strategy for confident sample selection
    # Two-level selection: (1) entity-level; (2) sample-level.
    parser.add_argument("--sample_level_selection", default=None, type=str, nargs="+", choices=["th_spl"], help="sample level selection method")
    parser.add_argument("--entity_level_selection", default=None, type=str, choices=["th_ent", "all", "TSMV", "MV"], help="entity level selection method")
    # hyper-parameters needed in selection method
    parser.add_argument("--confident_sample_size", default=None, type=int)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--sample_threshold", default=5, type=float, help="For th_spl, threshold in sample-level")
    parser.add_argument("--entity_threshold", default=5, type=float, help="For th_ent, threshold in entity-level")
    parser.add_argument("--first_level", default="entity", type=str, choices=["entity", "sample"], help="which level to firstly conduct selection")
    parser.add_argument("--emb", default=None, type=str, choices=["GPT", "SBERT"])

    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()
    
    # stop_ls = ["\n", "[]", "[{}]"]
    # stop_ls = ["[]", "[{}]"]
    stop_ls = None
    args.stop = stop_ls
    
    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
        args.demo_retrieval_method = None
    if args.few_shot_setting == "zs":
        args.few_shot_number = 0 
        args.demo_retrieval_method = None

    if args.reason_hint is None:
        args.reason_hint_pos = None
        args.reason_hint_person = None

    if args.tool_aug is None:
        args.tool_desc = None

    if not args.consistency:
        args.temperature = 0

    if args.consistency:
        assert args.temperature > 0
        assert args.query_times > 0
    else:
        assert args.temperature == 0

    args = get_paths(args)

    print("\n\n\n---------- Select Confident Annotations ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)