from typing import List, Dict
import json
import logging, logging.config
import os
import pandas as pd

from tqdm import tqdm
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import get_logger, load_data, save_data, assert_gpt35_turbo_16k, copy_file_to_path, compute_metrics, set_api_key
from utils_parse_answer import two_stage_majority_voting, majority_voting, parse_response_std

from const import dataset_language_map, model_list

logger = logging.getLogger()


def parse_response(args, data):
    data_response_parsed, data_w_all_SC_ans = parse_response_std(args, data)

    return data_response_parsed, data_w_all_SC_ans

def main(args):

    if args.self_annotation: # self-annotation mode
        if args.confident:
            data_response = load_data(args.pred_path)
        
            # compute metrics
            compute_metrics(args, data_response)
        else:
            # if need copying response file
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

            data_response = load_data(args.response_path)

            # compute metrics
            compute_metrics(args, data_response)

            for i in range(len(data_response)):
                data_response[i]["prediction"] = str(data_response[i]["prediction"])
                if args.consistency:
                    data_response[i]["responses"] = str(data_response[i]["responses"])
                    data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
                    data_response[i]["consistency_score_SC_all_ans"] = str(data_response[i]["consistency_score_SC_all_ans"])
                    data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
                else:
                    data_response[i]["response"] = str(data_response[i]["response"])
            save_data(args.pred_path, data_response)

    else: # self-supervision mode
        # if need copying response file
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

        data_response = load_data(args.response_path)

        # compute metrics
        compute_metrics(args, data_response)

        for i in range(len(data_response)):
            data_response[i]["prediction"] = str(data_response[i]["prediction"])
            if args.consistency:
                data_response[i]["responses"] = str(data_response[i]["responses"])
                data_response[i]["consistency_score"] = str(data_response[i]["consistency_score"])
                data_response[i]["consistency_score_SC_all_ans"] = str(data_response[i]["consistency_score_SC_all_ans"])
                data_response[i]["prediction_per_consist"] = str(data_response[i]["prediction_per_consist"])
            else:
                data_response[i]["response"] = str(data_response[i]["response"])
        save_data(args.pred_path, data_response)

    # if args.output_SC_all_answer == 1:
    #     data_w_all_SC_ans = collect_all_SC_answers(args, data_response)
    #     save_data(args.SC_all_ans_path, data_w_all_SC_ans)
    #     logger.info(f"Data with ALL SC answers saved to: {args.SC_all_ans_path}")


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
        if args.n_skip is not None:
            folder_0 = f"{folder_0}_skip_{args.n_skip}"
        if args.demo_retrieval_method in ["GPTEmbDvrsKNN"]:
            if args.diverseKNN_sampling in ["random", "Sc"]:
                diverse_knn_tag = f"{args.diverseKNN_number}_{args.diverseKNN_sampling}"
            elif args.diverseKNN_sampling in ["ScClEn"]:
                diverse_knn_tag = f"{args.diverseKNN_number}_{args.diverseKNN_sampling}_{args.weight_Sc}_{args.weight_Cl}_{args.weight_En}"
            else:
                raise ValueError(f"Unrecognized diverseKNN_sampling = {args.diverseKNN_sampling}")
            folder_0 = f"{folder_0}_{diverse_knn_tag}"
    if args.tool_aug:
        folder_0 = f"{folder_0}_tool"

    prompt_folder=folder_0
    if args.consistency: # add consistency tag
        flag_majority_voting_choices = {"two_stage_majority_voting":"TSMV", "majority_voting":"MV"}
        flag_majority_voting = flag_majority_voting_choices[args.consistency_selection]
        folder = f"{folder_0}_consist_{args.temperature}_{args.query_times}_{flag_majority_voting}"
        # Alternative response path for copying files
        folder_MV = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_MV"
        folder_TSMV = f"{folder_0}_consist_{args.temperature}_{args.consis_level}_{args.query_times}_TSMV"
        
        # SC voting method
        MV_func_choices = {"two_stage_majority_voting": two_stage_majority_voting,
                            "majority_voting": majority_voting}
        args.MV_func = MV_func_choices[args.consistency_selection]
    else:
        folder = folder_0

    # Add tags: self-consistent-annotate、self-annotation/self-supervision、demo_datamode (eg., train_shuflle_42)
    sca_folder = "self_consistent_annotate"
    sa_sp_folder = "self_annotation" if args.self_annotation else "self_supervision"
    datamode_folder = args.datamode if args.self_annotation else args.demo_datamode

    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    start_time = args.start_time
    datamode = args.datamode  

    model_folder = model_list[args.model]["abbr"]

    if args.self_annotation:
        if not args.confident:
            # ----- (1) all response and result metric files -----
            prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
            logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_Metr.log"
            response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
            pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
            # save all answers
            SC_ans_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_SC_all_ans.json"
            metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
            twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

            args.prompt_path = f"prompts/{sca_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{prompt_folder}/{prompt_filename}"
            args.response_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{response_filename}"
            args.response_dir = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}"
            args.pred_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{pred_filename}"
            # save all answers
            args.SC_all_ans_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{SC_ans_filename}"
            args.metric_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{metric_filename}"
            args.twostage_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{twostage_filename}"

            if args.consistency:
                # Backup response path, used to copy the response file when calculating different major voting methods
                args.response_MV_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder_MV}/{response_filename}"
                args.response_TSMV_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder_TSMV}/{response_filename}"
        else:
            # ----- (2) confident response and result metric files -----
            confident_prefix = f"{args.conf_select_method}"
            if args.confident_sample_size:
                confident_sample_size = f"_{args.confident_sample_size}"
            else:
                confident_sample_size = ""

            if args.confidence_scoring_method=="SC":
                logger_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{confident_prefix}_Metr.log"
                confident_response_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{confident_prefix}_response.txt"
                confident_pred_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{confident_prefix}_response.json"
                confident_metric_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{confident_prefix}{confident_sample_size}_metrics.csv"
                confident_twostage_filename = f"{start_time}_{datamode}_{prompt_method_name}_{args.few_shot_number}_{confident_prefix}{confident_sample_size}_metrics_twostage.csv"
            else:
                raise ValueError(f"Unrecognized confidence_scoring_method: {args.confidence_scoring_method}")

            args.response_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{confident_response_filename}"
            args.pred_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{confident_pred_filename}"
            args.metric_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{confident_metric_filename}"
            args.twostage_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{confident_twostage_filename}"
    else:
        # ----- Self-supervision -----
        # For self-supervision，add self-annotation tag
        st_prefix = ""
        if args.self_annotate_tag:
            st_prefix = f"_st_{args.self_annotate_tag}"

        prompt_filename = f"st_{args.self_annotate_tag}_{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
        logger_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_Metr.log"
        response_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.txt"
        pred_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_response.json"
        # save all answers
        SC_ans_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_SC_all_ans.json"
        metric_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics.csv"
        twostage_filename = f"{start_time}{st_prefix}_{datamode}_{prompt_method_name}_{args.few_shot_number}_metrics_twostage.csv"

        args.prompt_path = f"prompts/{sca_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{prompt_folder}/{prompt_filename}"
        args.response_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{response_filename}"
        args.response_dir = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}"
        args.pred_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{pred_filename}"
        # save all answers
        args.SC_all_ans_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{SC_ans_filename}"
        args.metric_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{metric_filename}"
        args.twostage_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}/{twostage_filename}"

        if args.consistency:
            # Backup response path, used to copy the response file when calculating different major voting methods
            args.response_MV_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder_MV}/{response_filename}"
            args.response_TSMV_path = f"result/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder_TSMV}/{response_filename}"

    # Logger setting
    log_dir = f"log/{sca_folder}/{model_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{folder}"
    args.log_path = os.path.join(log_dir, logger_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_dir = f"config"
    logger = get_logger(logger_filename, log_dir, config_dir)
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=str)
    parser.add_argument("--datamode", default=None, type=str)
    parser.add_argument("--demo_datamode", default=None, type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--ports", default=None, nargs="+", type=int)
    
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)
    # [None, key_noun, key_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="b", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="zs", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=1, type=int)
    parser.add_argument("--demo_select_method", default=None)
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=[None, "random", "GPTEmbCos", "GPTEmbDvrsKNN"])
    parser.add_argument("--few_shot_number", default=3, type=int)
    # skip tok-k in KNN
    parser.add_argument("--n_skip", default=None, type=int, help="skip top-n in Cosine Similar Retrieval.")
    # settings for diverseKNN
    parser.add_argument("--diverseKNN_number", default=50, type=int, help="#samples in diverse KNN.")
    parser.add_argument("--diverseKNN_sampling", default="random", type=str, choices=["random", "Sc", "ScClEn"], help="Sampling method to sample from diverseKNN")

    # self-consistency
    parser.add_argument("--consistency", default=0, type=int, choices=[0,1])
    parser.add_argument("--query_times", default=5, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--consis_level", default="question", type=str, choices=["question", "sample"])
    # SC voting method: [two_stage_majority_voting, ]
    parser.add_argument("--consistency_selection", default="two_stage_majority_voting", type=str)

    # Output all predicted answer groups, including those above the consistency threshold and those below the consistency threshold
    parser.add_argument("--output_SC_all_answer", default=0, type=int, choices=[0,1])

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int, choices=[0,1])
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    parser.add_argument("--start_time", default=None)

    # Two modes：1. self_supervision; 2. self_annotation
    parser.add_argument("--self_annotation", action="store_true")
    # For self-annotation, set to None; For self-supervision, set to tag
    parser.add_argument("--self_annotate_tag", default=None, type=str)

    # self_annotation: if use on confident sample；If so, set related hyparams
    parser.add_argument("--confident", action="store_true")
    # confidence score source：Self-Consistency, Self-verification
    parser.add_argument("--confidence_scoring_method", default="SC", type=str, choices=["SC","SV"])
    # confident sample selection strategt
    parser.add_argument("--conf_select_method", default=None, type=str)
    # consistent self-annotation
    # if 0, compute metrics on all confident sample; else，compute metrics on a specific number of confident samples
    parser.add_argument("--confident_sample_size", default=0, type=int)

    args = parser.parse_args()

    # prompt_pool，chinese/english
    args.lang = dataset_language_map[args.dataname]
    prompt_pool_choices = {
        "en": PromptPoolEnglish,
        "zh": PromptPoolChinese
    }
    args.prompt_pool = prompt_pool_choices[args.lang]
    
    stop_ls = None
    args.stop = stop_ls

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

    chat_paradigm="standard"
    if model_list[args.model]["publisher"] == "openai":
        # Change the model according to the maximum context length requirement
        assert_gpt35_turbo_16k(args, chat_paradigm=chat_paradigm)

        args.api_key = set_api_key(model_name=args.model, ports=args.ports)

    args = get_paths(args)
    
    logger.info("\n\n\n---------- Parse answers ------------")
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    main(args)