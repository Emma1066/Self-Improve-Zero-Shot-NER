import os
import json
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import traceback
import tiktoken
import argparse

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.dirname( path.abspath(__file__) ) ) ))

from utils import load_data, save_data, json2dict
from const import dataset_language_map, model_list

def compute_metric(data):
    n_gold = 0
    n_correct = 0
    n_pred = 0
    for _, item in enumerate(data):
        label = item["label"]
        if isinstance(label, str):
            label = eval(label)
        if isinstance(label, list):
            label = json2dict(label)
        prediction = item["self_annotate"]["prediction"]
        if isinstance(prediction, str):
            prediction = eval(prediction)
        if isinstance(prediction, list):
            prediction = json2dict(prediction)
        
        n_gold += len(label)
        n_pred += len(prediction)

        for pred_ment in prediction:
            if pred_ment in label and prediction[pred_ment] == label[pred_ment]:
                n_correct += 1

    prec = n_correct / n_pred if n_pred else 0
    rec = n_correct / n_gold if n_gold else 0
    if prec and rec:
        f1 = 2 * prec * rec / (prec+rec)
    else:
        f1 = 0
    
    prec = round(prec, 2)
    rec = round(rec, 2)
    f1 = round(f1, 2)

    return prec, rec, f1

def get_annotation_from_response(args, data_confident_pred, data, data_parse, data_GPTEmb):
    self_annotate_data = []
    self_annotate_data_parse = []
    self_annotate_data_GPTEmb = []
    assert args.self_annotate_tag
    self_annotate_tag = args.self_annotate_tag
    # The unique id of each sample
    all_uid = [x["idx"] for x in data]
    for i in tqdm(range(len(data_confident_pred)), desc="prediction-->annotation"):
        if args.confident_sample_size >0 and i >= args.confident_sample_size:
            break

        sample_confident_pred = data_confident_pred[i]
        uid = sample_confident_pred["idx"] # Unique id of the sample
        ind = all_uid.index(uid)

        sample = data[ind]
        sample["self_annotate"] = {"prediction": sample_confident_pred["prediction"]}
        if "consistency_score" in sample_confident_pred:
            sample["self_annotate"]["consistency_score"] = sample_confident_pred["consistency_score"]
        self_annotate_data.append(sample)
        if args.include_parse:
            sample_parse = data_parse[ind]
            sample_parse["self_annotate"] = {"prediction": sample_confident_pred["prediction"]}
            if "consistency_score" in sample_confident_pred:
                sample_parse["self_annotate"]["consistency_score"] = sample_confident_pred["consistency_score"]
            self_annotate_data_parse.append(sample_parse)
        if args.include_emb:
            self_annotate_data_GPTEmb.append(data_GPTEmb[ind])
    
    if args.include_emb:
        self_annotate_data_GPTEmb = np.stack(self_annotate_data_GPTEmb, axis=0)

    prec, rec, f1 = compute_metric(self_annotate_data)
    print(f"prec = {prec}, rec = {rec}, f1 = {f1}")

    return self_annotate_data, self_annotate_data_parse, self_annotate_data_GPTEmb


def prediction2annotation(args):
    # Load the original train set, find the corresponding sample based on the confident result, and store it in the confident annotation
    train_data = load_data(args.train_data_path)
    if args.include_parse:
        train_data_parse = load_data(args.train_data_parse_path)
        assert len(train_data) == len(train_data_parse)
    else:
        train_data_parse = None
    if args.include_emb:
        train_data_GPTEmb = np.load(args.train_GPTEmb_path)
        assert len(train_data) == len(train_data_GPTEmb)
    else:
        train_data_GPTEmb = None

    # load prediction data
    data_confident_pred = load_data(args.confident_pred_path)
    assert len(data_confident_pred) >= args.confident_sample_size

    self_annotated_data, self_annotated_data_parse, self_annotated_data_GPTEmb = get_annotation_from_response(args, data_confident_pred, train_data, train_data_parse, train_data_GPTEmb)

    # Save the updated prediction data
    save_data(args.self_annotated_data_path, self_annotated_data)
    print(f"len(self_annotated_data) = {len(self_annotated_data)}")

    if args.include_parse:
        save_data(args.self_annotated_data_parse_path, self_annotated_data_parse)
        print(f"len(self_annotated_data_parse) = {len(self_annotated_data_parse)}")
    if args.include_emb:
        np.save(args.self_annotated_data_GPTEmb_path, self_annotated_data_GPTEmb)
        print(f"shape(self_annotated_data_GPTEmb) = {self_annotated_data_GPTEmb.shape}")


def add_rationale(args):
    # Load prediction data
    data_confident_pred = load_data(args.confident_pred_path)
    assert len(data_confident_pred) >= args.confident_sample_size

    # load self-annotated data
    self_annotated_data = load_data(args.self_annotated_data_path)
    self_annotated_data_parse = load_data(args.self_annotated_data_parse_path)
    assert len(self_annotated_data) == args.confident_sample_size

    for i in range(args.confident_sample_size):
        rationale_for_gold_label = data_confident_pred[i]["label_rationale"]
        rationale_for_prediction = data_confident_pred[i]["prediction_rationale"]

        self_annotated_data[i]["label_rationale"] = rationale_for_gold_label
        self_annotated_data[i]["self_annotate"][f"prediction_rationale"] = rationale_for_prediction
        self_annotated_data_parse[i]["label_rationale"] = rationale_for_gold_label
        self_annotated_data_parse[i]["self_annotate"][f"prediction_rationale"] = rationale_for_prediction
    
    # Save self-annotated data with rational added
    save_data(args.self_annotated_data_path, self_annotated_data)
    print(f"len(self_annotated_data) = {len(self_annotated_data)}")

    save_data(args.self_annotated_data_parse_path, self_annotated_data_parse)
    print(f"len(self_annotated_data_parse) = {len(self_annotated_data_parse)}")


def main(args):
    # Convert prediction to annotation
    if not args.add_rationale:
        prediction2annotation(args)
    else:
        add_rationale(args)


def get_paths(args):
    dataname = args.dataname

    # label path
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    parse_postfix = args.parse_tool
    
    # train data loading path
    # 1. data; 2. data with parsing; 3. ChatGPT embs;
    train_data_filename = "train.json"
    args.train_data_path = f"data/{dataname}/{train_data_filename}"
    train_data_parse_filename = f"train_parse_{parse_postfix}.json"
    args.train_data_parse_path = f"data/{dataname}/{train_data_parse_filename}"    
    args.train_GPTEmb_path = f"data/{dataname}/train_GPTEmb.npy"    
    
    # response loading pth
    folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder = f"fs_{folder}"    
    if args.few_shot_setting in ["fixed", "pool"]:
        folder = f"{folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder = f"{folder}_{args.demo_retrieval_method}"
        if args.demo_retrieval_method in ["GPTEmbDvrsKNN"]:
            if args.diverseKNN_sampling in ["random", "Sc"]:
                diverse_knn_tag = f"{args.diverseKNN_number}_{args.diverseKNN_sampling}"
            elif args.diverseKNN_sampling in ["ScClEn"]:
                diverse_knn_tag = f"{args.diverseKNN_number}_{args.diverseKNN_sampling}_{args.weight_Sc}_{args.weight_Cl}_{args.weight_En}"
            else:
                raise ValueError(f"Unrecognized diverseKNN_sampling = {args.diverseKNN_sampling}")
            folder = f"{folder}_{diverse_knn_tag}"
    if args.tool_aug:
        folder = f"{folder}_tool"

    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"


    # Add self-consistent-annotate tag、self-annotation/self-supervision tag、demo_datamode tag (eg., train_shuflle_42)
    sca_folder = "self_consistent_annotate"
    sa_sp_folder = "self_annotation"
    datamode_folder = args.demo_datamode

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    demo_datamode = args.demo_datamode
    # Two modes: 1. Convert prediction to annotation; 2. Only supplement rational
    conf_select_method_postfix =  "" if args.conf_select_method is None else f"_{args.conf_select_method}"
    if not args.add_rationale:
        confident_pred_filename = f"{start_time}_{demo_datamode}_{prompt_method_name}_{args.few_shot_number}{conf_select_method_postfix}_response.json"
    else:
        confident_pred_filename = f"{start_time}_{demo_datamode}_{prompt_method_name}_{args.few_shot_number}{conf_select_method_postfix}_response_rationale.json"

    model_folder = model_list[args.model]["abbr"]
    args.confident_pred_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{confident_pred_filename}"
    
    # self-annotation data saving path: 
    # 1. data; 2. data with parsing; 3. ChatGPT embs;.
    demo_datamode = args.demo_datamode
    # Set whether the selected demo usage mode is fixed or pool
    demo_setting = args.demo_setting

    
    demo_folder = f"demo_{args.demo_setting}"
    model_folder = model_list[args.model]["abbr"]
    

    actual_sample_size = args.confident_sample_size
    # When args.confident_sample_size=0时， the actual self-annotated sample size is total sample sizes in the file
    if args.confident_sample_size==0:
        data_confident_pred = load_data(args.confident_pred_path)
        actual_sample_size = len(data_confident_pred)

    self_annotated_data_dir = f"data/{dataname}/{sca_folder}/{model_folder}/{demo_datamode}/{demo_folder}"
    if not os.path.isdir(self_annotated_data_dir):
        os.makedirs(self_annotated_data_dir)
    args.self_annotated_data_path = os.path.join(self_annotated_data_dir, f"{demo_datamode}_demo_{demo_setting}_{args.self_annotate_tag}{conf_select_method_postfix}_{actual_sample_size}.json")
    args.self_annotated_data_parse_path = os.path.join(self_annotated_data_dir, f"{demo_datamode}_demo_{demo_setting}_{args.self_annotate_tag}{conf_select_method_postfix}_{actual_sample_size}_parse_{parse_postfix}.json")
    args.self_annotated_data_GPTEmb_path = os.path.join(self_annotated_data_dir, f"{demo_datamode}_demo_{demo_setting}_{args.self_annotate_tag}{conf_select_method_postfix}_{actual_sample_size}_GPTEmb.npy")

    # Check if the corresponding self-annotated confident demo pool already exists
    if not args.add_rationale:
        if os.path.exists(args.self_annotated_data_path):
            raise ValueError(f"Demo file already exists: {args.self_annotated_data_path}")

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=int) # only for ace04
    # mode
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    # [None, key_noun, key_noun_verb, key_noun_verb_con_dep, key_con_dep_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=0, type=int)
    parser.add_argument("--demo_select_method", default=None) # "random", "GPTEmbClusterKmeans"
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=["random", "GPTEmbCos","GPTEmbDvrsKNN"])
    parser.add_argument("--few_shot_number", default=0, type=int)
    # settings for diverseKNN
    parser.add_argument("--diverseKNN_number", default=50, type=int, help="#samples in diverse KNN.")
    parser.add_argument("--diverseKNN_sampling", default="random", type=str, choices=["random", "Sc", "ScClEn"], help="Sampling method to sample from diverseKNN")

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy"])

    # self-annotation
    parser.add_argument("--demo_datamode", default="test", type=str)
    # choices: ["mpb_8, mpb_8_ClEn, th_entity_5, th_entity_3, th_sample_5, th_sample_2.5"]
    parser.add_argument("--conf_select_method", default=None, type=str)
    # set to 0 means using all samples
    parser.add_argument("--confident_sample_size", default=None, type=int)

    # self-training setting
    parser.add_argument("--self_annotate_tag", default=None, type=str) # basic, tool_aug, syn_prompt
    parser.add_argument("--demo_setting", default=None, type=str, choices=["pool", "fixed"])

    parser.add_argument("--include_parse", default=0, type=int, choices=[0,1])
    parser.add_argument("--include_emb", default=1, type=int, choices=[0,1])

    # if only adding rationale
    parser.add_argument("--add_rationale", action="store_true")

    parser.add_argument("--start_time", default=None)

    
    args = parser.parse_args()

    args.lang = dataset_language_map[args.dataname]

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

    args = get_paths(args)

    print("---------- Self-training: Response --> Annotation ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

