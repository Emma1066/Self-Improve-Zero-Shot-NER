import os
import json
import random
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import traceback
from openai.embeddings_utils import cosine_similarity
import tiktoken
import argparse

from DesignPrompts import PromptPoolChinese, PromptPoolEnglish

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import max_tokens, num_tokens_from_messages, load_data, save_data, assert_gpt35_turbo_16k
from const import dataset_language_map, model_list


def load_demo_data(path, demo_num):
        if demo_num:
            demo_data = load_data(path)
        else:
            demo_data = list()

        return demo_data

class PromptGenerator(object):
    def __init__(self, args) -> None:
        self.args = args
        self.dataname = args.dataname
        self.lanuage = dataset_language_map[self.dataname]
        prompt_pool_choices = {
            "en": PromptPoolEnglish,
            "zh": PromptPoolChinese
        }
        self.prompt_pool = prompt_pool_choices[self.lanuage](args.dataname)

    def retrieval_demo_by_emb(self, demo_data, n_demo, query_emb, demo_embs):
        demo_df = pd.DataFrame(columns=["embedding"])
        demo_df["embedding"] = list(demo_embs)
        # about 1.4s
        demo_df["similarity"] = demo_df.embedding.apply(lambda x: cosine_similarity(x, query_emb))
        
        cos_sims = demo_df["similarity"]
        sorted_idxes = np.argsort(cos_sims).tolist()
        sorted_idxes.reverse()
        # skip top-n
        if args.n_skip is not None:
            sorted_idxes = sorted_idxes[args.n_skip:]
        
        demos_selected = []
        cnt = 0
        while len(demos_selected) < n_demo:
            # Do not select samples without entity labels
            # if len(sorted_idxes[cnt]["label"]) == 0:
            #     continue
            demos_selected.append(demo_data[sorted_idxes[cnt]])
            cnt += 1

        # Put the more similar ones at the back
        demos_selected.reverse()

        return demos_selected

    def retrieval_diverseKNN_by_emb(self, demo_data, n_demo, query_emb, demo_embs):
        demo_df = pd.DataFrame(columns=["embedding"])
        demo_df["embedding"] = list(demo_embs)
        # about 1.4s
        demo_df["similarity"] = demo_df.embedding.apply(lambda x: cosine_similarity(x, query_emb))
        
        cos_sims = demo_df["similarity"]
        sorted_idxes = np.argsort(cos_sims).tolist()
        sorted_idxes.reverse()
        
        # Retrieval diverse KNN, K >> few_shot_number
        diverse_demos_selected = []
        cnt = 0
        while len(diverse_demos_selected) < self.args.diverseKNN_number:
            # Do not select samples without entity labels
            # if len(sorted_idxes[cnt]["label"]) == 0:
            #     continue
            diverse_demos_selected.append(demo_data[sorted_idxes[cnt]])
            cnt += 1
        
        # TODO: selection from diverse KNN
        demos_selected = []
        if self.args.diverseKNN_sampling == "random":
            demos_selected = self.retrieval_demo_by_random(diverse_demos_selected, n_demo=n_demo)
        elif self.args.diverseKNN_sampling == "Sc":
            demos_selected = self._retrieval_demo_by_sc_score(diverse_demos_selected, n_demo=n_demo)
        elif self.args.diverseKNN_sampling == "ScClEn":
            demos_selected = self._retrieval_demo_by_sc_score_ClEn(diverse_demos_selected, n_demo=n_demo)
        else:
            raise ValueError(f"Unrecognized diverseKNN_sampling = {args.diverseKNN_sampling}")

        # demos_selected.reverse()

        return demos_selected
    
    def _retrieval_demo_by_sc_score(self, demo_data, n_demo):
        sc_score_ls = []
        for item in demo_data:
            consistency_score = item["self_annotate"]["consistency_score"]
            if isinstance(consistency_score, str):
                consistency_score = eval(consistency_score)
            sc_score_ls.append(consistency_score["avg"])
        sc_score_ls = np.array(sc_score_ls)
        ranked_indexes = np.argsort(sc_score_ls).tolist()
        ranked_indexes.reverse()

        # TODO：debug
        # ranked_score = [sc_score_ls[x] for x in ranked_indexes]
        # print(f"\n{ranked_score}\n")

        selected_demos = []
        cnt = 0
        while len(selected_demos) <  n_demo:
            selected_demos.append(demo_data[ranked_indexes[cnt]])
            cnt += 1
        return selected_demos

    def _retrieval_demo_by_sc_score_ClEn(self, demo_data, n_demo):
        # collect all indexes
        consistency_score_all = []
        num_class_all = []
        num_entity_all = []
        for _, item in tqdm(enumerate(demo_data)):
            # consistency score
            tmp_consistency_score = item["self_annotate"]["consistency_score"]
            if isinstance(tmp_consistency_score, str):
                tmp_consistency_score = eval(tmp_consistency_score)
            consistency_score_all.append(tmp_consistency_score["avg"])
            # number of class
            tmp_prediction = item["self_annotate"]["prediction"]
            if isinstance(tmp_prediction, str):
                tmp_prediction = eval(tmp_prediction)
            num_class_all.append(len(set(list(tmp_prediction.values()))))
            # number of entity
            num_entity_all.append(len(tmp_prediction))

        # Normalize each index
        max_consistency_score =max(consistency_score_all)
        if max_consistency_score:
            consistency_score_all = [x/max_consistency_score for x in consistency_score_all]
        max_num_class = max(num_class_all)
        if max_num_class:
            num_class_all = [x/max_num_class for x in num_class_all]
        max_num_entity = max(num_entity_all)
        if max_num_entity:
            num_entity_all = [x/max_num_entity for x in num_entity_all]

        # TODO: debug
        # print(f"consistency_score_all:\n{consistency_score_all}\n")
        # print(f"num_class_all:\n{num_class_all}\n")
        # print(f"num_entity_all:\n{num_entity_all}\n")

        # Final composited index
        final_score = []
        for cs, nc, ne in zip(consistency_score_all, num_class_all, num_entity_all):
            final_score.append(args.weight_Sc*cs + args.weight_Cl*nc + args.weight_En*ne)

        # Rank by composited index
        sorted_idx = list(np.argsort(np.array(final_score)))
        sorted_idx.reverse() # higher --> lower

        # TODO：debug
        # ranked_score = [final_score[x] for x in sorted_idx]
        # print(f"\n{ranked_score}\n")

        selected_demos = []
        cnt = 0
        while len(selected_demos) < n_demo:
            selected_demos.append(demo_data[sorted_idx[cnt]])
            cnt += 1
        
        return selected_demos

    def retrieval_demo_by_random(self, demo_data, n_demo):
        demos_selected = []
        demos_idx = []
        while len(demos_selected) < n_demo:
            tmp_idx = random.choice(range(len(demo_data)))
            # Example not repeated
            if tmp_idx in demos_idx:
                continue

            # Do not select samples without entity labels
            # if len(demo_data[tmp_idx]["label"]) == 0:
            #     continue

            demos_selected.append(demo_data[tmp_idx])
            demos_idx.append(tmp_idx)

        return demos_selected 

    def retrieval_demo(self, query_sample, demo_data, query_emb=None, demo_embs=None):
        '''retrieval demo'''
        if self.args.few_shot_setting == "zs":
            return []
        if self.args.few_shot_setting == "fixed":
            return demo_data
        if self.args.demo_retrieval_method == "random":
            return self.retrieval_demo_by_random(demo_data, self.args.few_shot_number)
        elif self.args.demo_retrieval_method in ["GPTEmbCos", "SBERTEmbCos"]:
            return self.retrieval_demo_by_emb(demo_data, self.args.few_shot_number, query_emb, demo_embs)
        elif self.args.demo_retrieval_method in ["GPTEmbDvrsKNN"]:
            return self.retrieval_diverseKNN_by_emb(demo_data, self.args.few_shot_number, query_emb, demo_embs)
        else:
            raise ValueError(f"Wrong demo_retrieval_method={self.args.demo_retrieval_method}")

    def generate_prompt_per_query(
            self, 
            query_sample, 
            demo_data,
            query_emb=None,
            demo_embs=None
    ):
        '''Generate a prompt for a single query'''
        demos = self.retrieval_demo(query_sample, demo_data, query_emb=query_emb, demo_embs=demo_embs)
        prefix = self.prompt_pool.get_prompt_prefix(self.args)
        demos_prompts = []
        tmp_prompt = prefix
        exceed_max_len_flag = False
        for _, demo in enumerate(demos):
            demo_prompt = self.prompt_pool.get_prompt_for_demo(self.args, demo)
            
            # maximum length
            tmp_prompt += demo_prompt
            if num_tokens_from_messages(tmp_prompt) > max_tokens(self.args.model) - args.output_token_len:
                print("\nExceed max len:\nidx = {}, sentence = \n{}".format(query_sample["idx"], query_sample["sentence"]))
                exceed_max_len_flag = True
                break

            demos_prompts.append(demo_prompt)

        demos_prompts = "".join(demos_prompts)
        postfix = self.prompt_pool.get_prompt_postfix(self.args, query_sample)

        prompt = prefix + demos_prompts + postfix
        
        if exceed_max_len_flag:
            print(prompt)

        return prompt, exceed_max_len_flag

    def generate_prompt_batch(
            self, 
            query_data, 
            demo_data,
            query_embs=None,
            demo_embs=None
    ):
        '''Generate the entire dataset prompt'''
        data_prompts = []
        exceed_max_len_cnt = 0
        if args.self_annotation:
            print(f"Annotation size = {args.annotation_size}")
            bar = tqdm(query_data[:args.annotation_size], ncols=100)
        else:
            bar = tqdm(query_data, ncols=100)
        # for i_query, query in enumerate(tqdm(query_data, desc="generate prompt", ncols=100)):
        for i_query, query in enumerate(bar):
            query_emb = None
            if self.args.demo_retrieval_method in ["GPTEmbCos", "SBERTEmbCos", "GPTEmbDvrsKNN"] and self.args.few_shot_setting in ["full","pool"]:
                assert query_embs is not None
                query_emb = query_embs[i_query]
            prompt, exceed_max_len_flag = self.generate_prompt_per_query(
                query, 
                demo_data,
                query_emb=query_emb,
                demo_embs=demo_embs
            )

            if exceed_max_len_flag:
                exceed_max_len_cnt += 1
                
            query_prompt = {
                "idx": query["idx"],
                "sentence": query["sentence"],
                "label": query["label"],
                "prompt": prompt
            }
            data_prompts.append(query_prompt)
        
        print(f"\nNumber of samples exceeding length limit = {exceed_max_len_cnt}")

        return data_prompts
    
def remove_annotated_sample(query_data, annotated_data, query_embs=None):
    remained_query_data = []
    if query_embs is not None:
        remained_query_embs = []
    else:
        remained_query_embs = None
    idxes_annotated = [x["idx"] for x in annotated_data]
    for i_item, item in enumerate(query_data):
        if item["idx"] in idxes_annotated:
            continue
        remained_query_data.append(item)
        if query_embs is not None:
            remained_query_embs.append(query_embs[i_item])
    
    print(f"# len(original) = {len(query_data)}")
    print(f"# len(annotated) = {len(annotated_data)}")
    print(f"# len(remained) = {len(remained_query_data)}")

    if query_embs is not None:
        remained_query_embs = np.stack(remained_query_embs, axis=0)
        print(f"# shape(original) = {len(query_embs)}")
        print(f"# shape(remained) = {len(remained_query_embs)}")

    return remained_query_data, remained_query_embs

def main(args):
    query_data = load_data(args.query_data_path)
    demo_data = load_demo_data(args.demo_data_path, demo_num=args.few_shot_number)

    if args.few_shot_setting == "fixed":
        assert len(demo_data) == args.few_shot_number

    # load embedding
    if args.demo_retrieval_method in ["GPTEmbCos", "SBERTEmbCos", "GPTEmbDvrsKNN"] and args.few_shot_setting in ["pool", "full"]:
        query_embs = np.load(args.query_embs_path)
        demo_embs = np.load(args.demo_embs_path)

        # TODO: debug
        print(len(query_data))
        print(query_embs.shape)

        assert len(query_data) == len(query_embs)
        assert len(demo_data) == len(demo_embs)
    else:
        query_embs = None
        demo_embs = None

    # In Iterative self annotate mode, remove the selected confidential sample demo from query_data
    if args.iterative_self_annotate:
        query_data, query_embs = remove_annotated_sample(query_data, annotated_data=demo_data, query_embs=query_embs)
    
    # generate prompt
    prompt_generator = PromptGenerator(args=args)
    prompts = prompt_generator.generate_prompt_batch(
        query_data, 
        demo_data,
        query_embs=query_embs,
        demo_embs=demo_embs
    )

    # lines or whole datasets as one json object
    json_format = None
    model_publisher = model_list[args.model]["publisher"]
    if model_publisher in ["meta"]:
        json_format="lines"
    save_data(args.save_prompt_path, prompts, json_format=json_format)


def get_paths(args):
    dataname = args.dataname

    # label path
    args.abb2labelname_path = f"data/{args.dataname}/abb2labelname.json"
    args.abb2lname = json.load(open(args.abb2labelname_path, "r", encoding="utf-8"))
    args.id2label = list(args.abb2lname.values())

    parse_postfix = args.parse_tool
    
    # embedding method
    if args.demo_retrieval_method:
        if "SBERTEmb" in args.demo_retrieval_method:
            emb = "SBERTEmb"
        elif "GPTEmb" in args.demo_retrieval_method:
            emb = "GPTEmb"
        else:
            emb = None
    elif args.demo_select_method:
        if "GPTEmb" in args.demo_select_method:
            emb = "GPTEmb"
        elif "SBERT" in args.demo_select_method:
            emb = "SBERT"
        else:
            emb = None
    
    # query data path
    datamode = args.datamode
    args.query_data_path = f"data/{dataname}/{datamode}.json"
    if args.tool_aug:
        args.query_data_path = f"data/{dataname}/{datamode}_parse_{parse_postfix}.json"
    args.query_embs_path = f"data/{dataname}/{datamode}_{emb}.npy"
    
    # demo data path
    if args.few_shot_setting == "zs":
        args.demo_data_path = None
    elif args.few_shot_setting == "full":
        demo_filename = "train.json"
        if args.reason_hint:
            if "pos" in args.reason_hint or "dep" in args.reason_hint or "con" in args.reason_hint or "tok" in args.reason_hint:
                demo_filename = f"train_parse_{parse_postfix}.json"
        elif args.tool_aug:
            if "Pos" in args.tool_aug or "Dep" in args.tool_aug or "Con" in args.tool_aug or "Tok" in args.tool_aug:
                demo_filename = f"train_parse_{parse_postfix}.json"
        args.demo_data_path = f"data/{dataname}/{demo_filename}"
        args.demo_embs_path = f"data/{dataname}/train_{emb}.npy"          

    elif args.few_shot_setting in ["fixed", "pool"]:
        demo_datamode = args.demo_datamode
        
        demo_folder = f"demo_{args.few_shot_setting}"
        model_folder = model_list[args.model]["abbr"]
        # If it is in self supervision mode, add a self annotate tag to the demo path
        if args.self_annotate_tag:
            demo_folder = os.path.join("self_consistent_annotate", model_folder, demo_datamode, demo_folder)

        demo_filename = f"{demo_datamode}_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}.json"
        if args.reason_hint:
            if "pos" in args.reason_hint or "dep" in args.reason_hint or "con" in args.reason_hint or "tok" in args.reason_hint:
                demo_filename = f"{demo_datamode}_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"
        elif args.tool_aug:
            if "Pos" in args.tool_aug or "Dep" in args.tool_aug or "Con" in args.tool_aug or "Tok" in args.tool_aug:
                demo_filename = f"{demo_datamode}_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_parse_{parse_postfix}.json"            
            
        args.demo_data_path = f"data/{dataname}/{demo_folder}/{demo_filename}"

        demo_embs_filename = f"{demo_datamode}_demo_{args.few_shot_setting}_{args.demo_select_method}_{args.demo_size}_{emb}.npy"
        args.demo_embs_path = f"data/{dataname}/{demo_folder}/{demo_embs_filename}"          

    else:
        raise ValueError(f"Wrong few_shot_setting = {args.few_shot_setting}")
  
    # prompt saving path
    prompt_folder = f"{args.few_shot_setting}"
    if args.few_shot_setting in ["fixed", "pool", "full"]:
        prompt_folder = f"fs_{prompt_folder}"
    if args.few_shot_setting in ["fixed", "pool"]:
        prompt_folder = f"{prompt_folder}_{args.demo_select_method}_{args.demo_size}"
    if args.few_shot_setting in ["pool", "full"]:
        prompt_folder = f"{prompt_folder}_{args.demo_retrieval_method}"
        if args.n_skip is not None:
            prompt_folder = f"{prompt_folder}_skip_{args.n_skip}"
        if args.demo_retrieval_method in ["GPTEmbDvrsKNN"]:
            if args.diverseKNN_sampling in ["random", "Sc"]:
                diverse_knn_tag = f"{args.diverseKNN_number}_{args.diverseKNN_sampling}"
            else:
                raise ValueError(f"Unrecognized diverseKNN_sampling = {args.diverseKNN_sampling}")
            prompt_folder = f"{prompt_folder}_{diverse_knn_tag}"
    if args.tool_aug:
        prompt_folder = f"{prompt_folder}_tool"


    tool_aug = args.tool_aug
    if args.tool_aug and args.tool_desc:
        tool_aug  = f"{tool_aug}Desc"
    prompt_tricks = [args.task_hint, args.reason_hint, args.reason_hint_person, args.reason_hint_pos, tool_aug]
    prompt_tricks = [x for x in prompt_tricks if x]
    prompt_method_name = "_".join(prompt_tricks)

    if args.self_annotation: # self-annotate
        prompt_filename = f"{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
    else: # self-supervision
        prompt_filename = f"st_{args.self_annotate_tag}_{datamode}_prompts_{prompt_method_name}_{args.few_shot_number}.json"
 
    sca_folder = "self_consistent_annotate"
    sa_sp_folder = "self_annotation" if args.self_annotation else "self_supervision"
    datamode_folder = datamode if args.self_annotation else args.demo_datamode

    prompt_dir = f"prompts/{sca_folder}/{dataname}/{sa_sp_folder}/{datamode_folder}/{prompt_folder}"
    if not os.path.exists(prompt_dir):
        os.makedirs(prompt_dir)
    args.save_prompt_path = os.path.join(prompt_dir, prompt_filename)

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default="PowerPlantFlat", type=str)
    parser.add_argument("--folder", default=0, type=int) # only for ace04
    parser.add_argument("--datamode", default="test", type=str)
    parser.add_argument("--demo_datamode", default=None, type=str)
    # model
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--output_token_len", default=1000, type=int)
    # prompt
    # parser.add_argument("--prompt_method", default="vanilla")
    parser.add_argument("--task_hint", default=None)

    # [None, key_noun, key_noun_verb, key_noun_verb_con_dep, key_con_dep_noun_verb]
    parser.add_argument("--reason_hint", default=None)
    parser.add_argument("--reason_hint_person", default="first", choices=["first", "second"])
    parser.add_argument("--reason_hint_pos", default="f", choices=[None, "f", "b"])
    # retrieval
    parser.add_argument("--few_shot_setting", default="pool", choices=["fixed", "pool", "full", "zs"])
    parser.add_argument("--demo_size", default=300, type=int)
    parser.add_argument("--demo_select_method", default="GPTEmbClusterKmeans")
    parser.add_argument("--demo_retrieval_method", default="GPTEmbCos", choices=["random", "GPTEmbCos", "SBERTEmbCos", "GPTEmbDvrsKNN"])
    parser.add_argument("--few_shot_number", default=1, type=int)
    # skip tok-k in KNN
    parser.add_argument("--n_skip", default=None, type=int, help="skip top-n in Cosine Similar Retrieval.")
    # settings for diverseKNN
    parser.add_argument("--diverseKNN_number", default=50, type=int, help="#samples in diverse KNN.")
    parser.add_argument("--diverseKNN_sampling", default="random", type=str, choices=["random", "Sc", "ScClEn"], help="Sampling method to sample from diverseKNN")

    # tool aug
    parser.add_argument("--tool_aug", default=None, choices=[None, "ToolTokCoarse", "ToolPos", "ToolDep", "ToolCon"])
    parser.add_argument("--tool_desc", default=1, type=int)
    parser.add_argument("--parse_tool", default="hanlp", choices=["hanlp", "spacy", "stanza"])

    # Two modes：1. self_supervision; 2. self_annotation
    parser.add_argument("--self_annotation", action="store_true")    
    # For self-annotation, set to None; For self-supervision set to corresponding tag.
    parser.add_argument("--self_annotate_tag", default=None, type=str) # basic, tool_aug, syn_prompt, ToolDep_ToolUseHint_first_b_consist_5_confident
    # Iterative self-annotating
    parser.add_argument("--annotation_size", default=None, type=int)
    parser.add_argument("--iterative_self_annotate", action="store_true")
        
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
    if args.few_shot_setting != "zs":
        if args.reason_hint:
            assert args.reason_hint_person == "second"
            assert args.reason_hint_pos == "f"

    # Change the model according to the maximum context length requirement
    assert_gpt35_turbo_16k(args, chat_paradigm="standard")
   
    args = get_paths(args)

    print("---------- Generate prompts ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

