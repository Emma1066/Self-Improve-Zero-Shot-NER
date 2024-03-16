import json
import time
import logging, logging.config
import os
import re
import pandas as pd
from collections import Counter

from tqdm import tqdm
import argparse

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import get_logger, load_data, save_data, json2dict, copy_file_to_path, compute_metrics
from utils_parse_answer import response_2_prediction, two_stage_majority_voting, majority_voting, collect_all_SC_answers, compute_consistency_score, parse_response_std
from const import model_list

logger = logging.getLogger()


def main(args):
    # Determine if it is necessary to copy the response file
    if not os.path.exists(args.response_path):
        if args.consistency_selection == "two_stage_majority_voting":
            copying_file_path = args.response_MV_path
        elif args.consistency_selection == "majority_voting":
            copying_file_path = args.response_TSMV_path
        else:
            raise ValueError(f"Unrecognized consistency selection: {args.consistency_selection}")
        copying_data = load_data(copying_file_path)
        copy_file_to_path(copying_data, args.response_dir, args.response_path)
        logger.info(f"File is copied to: {args.response_path}")

    # load data
    data_response = load_data(args.response_path)

    compute_metrics(args, data_response)
    
    for i in range(len(data_response)):
        data_response[i]["prediction"] = str(data_response[i]["prediction"])
        if args.consistency:
            data_response[i]["responses"] = str(data_response[i]["responses"])
            data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
            data_response[i]["consistency_score_SC_all_ans"] = str(data_response[i]["consistency_score_SC_all_ans"])
    save_data(args.pred_path, data_response)
    logger.info(f"Prediction data saved to: {args.pred_path}")


def get_paths(args):
    dataname = args.dataname

    # label path
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    # response path
    folder_0 = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        folder_0 = f"fs_{folder_0}"    
    if args.few_shot_setting in ["fixed", "pool"]:
        folder_0 = f"{folder_0}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        folder_0 = f"{folder_0}_{args.demo_retrieval_method}"
    if args.tool_aug:
        folder_0 = f"{folder_0}_tool"
    if args.consistency:
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"
        # Alternative response path for copying files
        folder_MV = f"{folder_0}_consist_{args.temperature}_{args.query_times}_MV"
        folder_TSMV = f"{folder_0}_consist_{args.temperature}_{args.query_times}_TSMV"

        # SC voting method
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting,
                            "majority_voting": majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]
    else:
        folder = folder_0

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    datamode = args.datamode
    response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
    pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
    # Store all SC answers
    SC_ans_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_SC_all_ans.json"
    logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_ParseAns.log"
    metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
    twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

    model_folder = model_list[args.model]["abbr"]

    args.response_path = f"result/{model_folder}/{dataname}/{folder}/{response_filename}"
    args.response_dir = f"result/{model_folder}/{dataname}/{folder}"
    args.pred_path = f"result/{model_folder}/{dataname}/{folder}/{pred_filename}"
    # Store all SC answers
    args.SC_all_ans_path = f"result/{model_folder}/{dataname}/{folder}/{SC_ans_filename}"
    args.metric_path = f"result/{model_folder}/{dataname}/{folder}/{metric_filename}"
    args.twostage_path = f"result/{model_folder}/{dataname}/{folder}/{twostage_filename}"
    
    # Backup response path, used to copy the response file when calculating different major voting methods
    if args.consistency==1:
        args.response_MV_path = f"result/{model_folder}/{dataname}/{folder_MV}/{response_filename}"
        args.response_TSMV_path = f"result/{model_folder}/{dataname}/{folder_TSMV}/{response_filename}"

    # Logger setting
    log_dir = f"log/{model_folder}/{dataname}/{folder}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default=None, type=str)
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--datamode", default="test", type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    # prompt
    parser.add_argument("--task_hint", default=None)
    # [None, key_noun, key_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=1, type=int)
    # choices=["random_42", "GPTEmbClusterKmeans"]
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos", "SBERTEmbCos"])
    parser.add_argument("--few_shot_number", default=3, type=int)

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    # SC voting method: [two_stage_majority_voting, ]
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # Output all predicted answer groups, including those above the consistency threshold and those below the consistency threshold
    parser.add_argument("--output_SC_all_answer", default=0, type=int, choices=[0,1])

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1)
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    parser.add_argument("--start_time", default=None)

    args = parser.parse_args()

    
    assert args.start_time is not None
    if args.few_shot_setting == "fixed":
        args.few_shot_number = args.demo_size
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

    logger.info("\n\n\n---------- Parse answers ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)