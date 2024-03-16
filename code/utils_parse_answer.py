import logging, logging.config
from typing import List, Dict
from argparse import Namespace
import re
from tqdm import tqdm
from utils import json2dict, convert_format

logger = logging.getLogger()

def parse_response_std(args, data):
    data_response_parsed = []
    data_with_all_SC_ans = []
    for i_item, item in enumerate(tqdm(data, desc="parse responses")):
        if args.consistency == 0:
            response = item["response"]
            prediction = response_2_prediction(args, item, response)
            item["prediction"] = prediction
        else:
            # SC voting method
            MV_func = args.MV_func
            responses = item["responses"]
            if isinstance(responses, str):
                responses = eval(responses)
            assert isinstance(responses, list)
            prediction_per_consist = [response_2_prediction(args, item, tmp_resp) for tmp_resp in responses]
            item["prediction_per_consist"] = prediction_per_consist

            # if args.consistency_selection == "two_stage_majority_voting":
            #     prediction = two_stage_majority_voting(args, prediction_per_consist)
            # elif args.consistency_selection == "majority_voting":
            #     prediction = majority_voting(args, prediction_per_consist)
            prediction = MV_func(args, prediction_per_consist)
            item["prediction"] = prediction
            # compute voted answers' score
            consistency_score_entities = compute_consistency_score(prediction_per_consist, voted_prediction=prediction)
            if len(consistency_score_entities):
                consistency_score_avg = sum(list(consistency_score_entities.values())) / len(consistency_score_entities)
            else:
                consistency_score_avg = 0
            item["consistency_score"] = {"entities": consistency_score_entities, "avg":consistency_score_avg} # consistenct score of the final voted entities (dict)
            # compute all answers' score
            consistency_score_SC_all_ans = compute_consistency_score(prediction_per_consist, voted_prediction=None)
            if len(consistency_score_SC_all_ans):
                consistency_score_SC_all_ans_avg = sum(list(consistency_score_SC_all_ans.values())) / len(consistency_score_SC_all_ans)
            else:
                consistency_score_SC_all_ans_avg = 0
            item["consistency_score_SC_all_ans"] = {"entities": consistency_score_SC_all_ans, "avg":consistency_score_SC_all_ans_avg}

            # output all answers
            if args.output_SC_all_answer==1:
                item_w_all_SC_ans = collect_all_SC_answers(args, item, prediction_tuple_count=consistency_score_SC_all_ans)
                data_with_all_SC_ans.append(item_w_all_SC_ans)

        data_response_parsed.append(item)
    
    return data_response_parsed, data_with_all_SC_ans

def collect_all_SC_answers(args, item, prediction_tuple_count):
    assert args.consistency==1

    if hasattr(args, "order") and args.order != None:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_quest", "prediction"]
    else:
        copying_keys = ["idx", "sentence", "label", "responses", "prediction_per_consist", "prediction"]
    
    # convert (mention, type) frequency into json
    label = item["label"]
    if isinstance(label, str):
        label = eval(label)
    if isinstance(label, list):
        label = json2dict(label)
    prediction_voted = item["prediction"]
    if isinstance(prediction_voted, str):
        prediction_voted = eval(prediction_voted)
    if isinstance(prediction_voted, list):
        prediction_voted = json2dict(prediction_voted)

    prediction_all_with_cnt = []
    prediction_correct_with_cnt = []
    prediction_wrong_with_cnt = []
    prediction_voted_with_cnt = []
    for (k,v), cnt in prediction_tuple_count.items():
        prediction_all_with_cnt.append(str({k:v, "SC Count": cnt}))
        if k in label and label[k]==v:
            prediction_correct_with_cnt.append(str({k:v, "SC Count": cnt}))
        else:
            prediction_wrong_with_cnt.append(str({k:v, "SC Count": cnt}))
        if k in prediction_voted and prediction_voted[k]==v:
            prediction_voted_with_cnt.append(str({k:v, "SC Count": cnt}))

    item_w_SC_all_ans = {}
    for k in copying_keys:
        item_w_SC_all_ans[k] = item[k]

    item_w_SC_all_ans["SC_all_ans"] = prediction_all_with_cnt
    item_w_SC_all_ans["SC_correct_ans"] = prediction_correct_with_cnt
    item_w_SC_all_ans["SC_wrong_ans"] = prediction_wrong_with_cnt
    item_w_SC_all_ans["SC_voted_ans"] = prediction_voted_with_cnt

    return item_w_SC_all_ans

def response_2_prediction_of_list(args, query, response, resp_idx=None, question=None):
    '''
    Returns: 
        predictions: list
    '''
    if response in ["", "[]", "[{}]"]:
        return []

    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]
    label_order =args.label_order
    target_type = label_order[resp_idx]
    if isinstance(target_type, list):
        if len(target_type) > 1:
            raise ValueError(f"target type is more than one: {len(target_type)}")
        target_type = target_type[0]

    punc_zh2en = {'，': ',', '。': '.', '：': ':'} # transfer Chinese punctuation into English punctuation 
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    response_punctransed = response_punctransed.replace("\n", "")
    matched_list = re.findall(r'\[(.*?)\]', response_punctransed) # matching
    if len(matched_list) == 0:
        logger.info(f"===== Error occured (Wrong Format): {sentid}")
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info("        Question: {}".format(question))
        logger.info(f"        Error response_{resp_idx}: \n{response}")
        logger.info("        Set and_processed as empty dict.")
        prediction = []
    else:
        try:
            ans_str = matched_list[-1] # normally the final answer appeared at the end of a reasoning process
            if "\"" in ans_str:
                ans_str = "[" + ans_str + "]"
                prediction = eval(ans_str)
            elif ans_str == "":
                prediction = []
            else:
                ans_ls_raw = ans_str.split(",")
                prediction = [x.strip() for x in ans_ls_raw] # remove possible blank space
            
        except Exception as e:
            logger.info(f"===== Error occured (Wrong Format): {sentid}")
            logger.info("        Sent: {}".format(sent))
            logger.info("        Label: {}".format(label))
            logger.info("        Question: {}".format(question))
            logger.info(f"        Error response_{resp_idx}: \n{response}")
            logger.info("        Set and_processed as empty dict.")
            logger.info(f"        Error traceback:")
            logger.info(str(e))    
            prediction = []
    
    return prediction    

def response_2_prediction_of_dict_json(args, query, response, resp_idx=None, question=None, return_form="dict"):
    if response in ["", "[]", "[{}]", "A: []"]:
        prediction = [] if return_form=="json" else {}
        return prediction

    sentid = query["idx"]
    sent = query["sentence"]
    label = query["label"]
    id2label =args.id2label

    punc_zh2en = {'，': ',', '。': '.', '：': ':'} # transfer Chinese punctuation into English punctuation 
    response_punctransed = response.translate(str.maketrans(punc_zh2en))
    response_punctransed = response_punctransed.replace("\n", "")
    matched_list = re.findall(r'\[(.*?)\]', response_punctransed) # matching
    if len(matched_list) == 0:
        # if not matched json
        # try matching dict
        if args.few_shot_setting == "zs":
            matched_list = re.findall(r'\{(.*?)\}', response_punctransed)
            prediction = []
            for matched_item in matched_list:
                matched_item = "{" + matched_item + "}"
                # null --> \"O\"
                matched_item = matched_item.replace("null", "\"O\"")
                eval_matched_item = eval(matched_item)
                if isinstance(eval_matched_item, dict):
                    for k, v in eval_matched_item.items():
                        if k in sent and v in id2label:
                            prediction.append({k:v})

            if len(prediction)>0:
                if return_form=="dict":
                    prediction=json2dict(prediction)
                return prediction


        logger.info(f"===== Error occured (No matched): {sentid}")
        logger.info("        Sent: {}".format(sent))
        logger.info("        Label: {}".format(label))
        logger.info("        Question: {}".format(question))
        logger.info(f"        Error response_{resp_idx}: \n{response}")
        logger.info("        Set and processed as empty dict.")
        prediction = []
    else:
        try:
            ans_str = '[' + matched_list[-1] + ']' # normally the final answer appeared at the end of a reasoning process
            # null  --> \"O\"
            ans_str = ans_str.replace("null", "\"O\"")
            
            ans_eval = eval(ans_str)

            if len(ans_eval)==0: # if returned empty list
                prediction = ans_eval
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction

            # deal with format：[{"Entity Name": "Oleg Shatskiku", "Entity Label": "PERSON"}, ...]
            # 1. English
            if "Entity Name" in ans_eval[0] and "Entity Label" in ans_eval[0]:
                prediction = []
                for tmp in ans_eval:
                    if len(tmp) == 0:
                        continue
                    if tmp["Entity Name"] in id2label: # if type and mention are reversed
                        tmp_ment = tmp["Entity Label"]
                        tmp_type = tmp["Entity Name"]
                    else:
                        tmp_ment = tmp["Entity Name"] 
                        tmp_type = tmp["Entity Label"] 
                    prediction.append({tmp_ment:tmp_type})
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction
            
            # 2. Chinese
            if "实体名称" in ans_eval[0] and "实体标签" in ans_eval[0]:
                prediction = []
                for tmp in ans_eval:
                    if tmp["实体名称"] in id2label: # if type and mention are reversed
                        tmp_ment = tmp["实体标签"] 
                        tmp_type = tmp["实体名称"]
                    else:
                        tmp_ment = tmp["实体名称"]
                        tmp_type = tmp["实体标签"] 
                    prediction.append({tmp_ment:tmp_type})
                if return_form=="dict":
                    prediction = json2dict(prediction)
                return prediction
            
            # deal with two possible format：
            # 1： [{XX:XX, XX:XX, XX:XX}]
            # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
            
            if len(ans_eval) == 1 and len(ans_eval[0]) > 1: # 1： [{XX:XX, XX:XX, XX:XX}]
                prediction_w_o = [
                    {k: v} for k,v in ans_eval[0].items()
                ]
            else: # 2： [{XX:XX}, {XX:XX}, {XX:XX}]
                # prediction_w_o = {list(item.keys())[0]: list(item.values())[0] for item in ans_eval}
                prediction_w_o = ans_eval
            # remove type of "O" (null)
            prediction = []
            for item in prediction_w_o:
                k, v = list(item.items())[0]
                if v != "O":
                    prediction.append(item)
        except Exception as e:
            logger.info(f"===== Error occured (Unparsable): {sentid}")
            logger.info("        Sent: {}".format(sent))
            logger.info("        Label: {}".format(label))
            logger.info("        Question: {}".format(question))
            logger.info(f"        Error response_{resp_idx}: \n{response}")
            logger.info("        Set and_processed as empty dict.")
            logger.info(f"        Error traceback:")
            logger.info(str(e))    
            prediction = []
    
    if return_form=="dict":
        prediction=json2dict(prediction)
    return prediction


def response_2_prediction(args, query, response, resp_idx=None, question=None, return_form="dict", complete_form="question", return_responded_qa=False):
    if complete_form == "question": # need model to answer answer of each question
        if return_form in ["dict", "json"]:
            prediction = response_2_prediction_of_dict_json(args, query, response, resp_idx=resp_idx, question=question, return_form=return_form)
        elif return_form == "list":
            prediction = response_2_prediction_of_list(args, query, response, resp_idx=resp_idx, question=question)
        else:
            raise ValueError(f"Unrecognized return_form: {return_form}")
        return prediction
    else:
        raise ValueError(f"Unrecognized complete_form={complete_form}")

def two_stage_majority_voting(args, prediction_ls=None, mention_type_cnt_all=None):
    '''
    Vote for most consistent named entities from a set of predictions.
    Two-stage voting: 1) entity mention; 2) entity type.
    Params:
        prediction_ls: list of prediction (dict);
    Returns:
        prediction_voted: voted prediction (dict)
    '''
    if isinstance(args, Namespace):
        tot_votes = args.query_times
    elif isinstance(args, int):
        tot_votes = args
    else:
        raise TypeError(f"Unknown type of args: {type(args)}")
    lowest_votes_for_O = tot_votes // 2
    if tot_votes % 2 == 1:
        lowest_votes_for_O += 1

    if mention_type_cnt_all is None:
        assert prediction_ls is not None
        mentions_all = []
        types_all = []
        for tmp_pred in prediction_ls:
            # convert json format to dict
            if isinstance(tmp_pred, list):
                tmp_pred = json2dict(tmp_pred)
            mentions_all += list(tmp_pred.keys())
            types_all += list(tmp_pred.values())

        mention_type_cnt_all = {}
        for tmp_mention, tmp_type in zip(mentions_all, types_all):
            if tmp_mention not in mention_type_cnt_all:
                mention_type_cnt_all[tmp_mention] = {}
            if tmp_type not in mention_type_cnt_all[tmp_mention]:
                mention_type_cnt_all[tmp_mention][tmp_type] = 0
            mention_type_cnt_all[tmp_mention][tmp_type] += 1
        
    # mentions_all_cnt = Counter(mentions_all)
    mentions_all_cnt = {}
    for k,v in mention_type_cnt_all.items():
        mentions_all_cnt[k] = sum(list(v.values()))
    voted_mentions = []
    for tmp_mention in mentions_all_cnt:
        if mentions_all_cnt[tmp_mention] >= lowest_votes_for_O:
            voted_mentions.append(tmp_mention)

    prediction_voted = {}
    for tmp_mention in voted_mentions:
        tmp_type_cnt = mention_type_cnt_all[tmp_mention]
        tmp_type_cnt = list(sorted(list(tmp_type_cnt.items()), key=lambda x: x[1], reverse=True))
        tmp_majority_type, tmp_majority_type_votes = tmp_type_cnt[0][0], tmp_type_cnt[0][1]

        prediction_voted[tmp_mention] = tmp_majority_type

    return prediction_voted

def majority_voting(args, prediction_ls=None, cnt_prediction_tuple=None):
    '''
    Vote for most consistent named entities from a set of predictions.
    Params:
        prediction_ls: list of prediction (dict);
    Returns:
        prediction_voted: voted prediction (dict)
    '''
    if isinstance(args, Namespace):
        tot_votes = args.query_times
    elif isinstance(args, int):
        tot_votes = args
    else:
        raise TypeError(f"Unknown type of args: {type(args)}")
    lowest_votes_for_O = tot_votes // 2
    if tot_votes % 2 == 1:
        lowest_votes_for_O += 1
    
    if cnt_prediction_tuple is None:
        assert prediction_ls is not None
        # count (mention, type) pairs
        cnt_prediction_tuple = {}
        for prediction in prediction_ls:
            if isinstance(prediction, list):
                prediction = json2dict(prediction)
            for k, v in prediction.items():
                if (k,v) not in cnt_prediction_tuple:
                    cnt_prediction_tuple[(k,v)] = 0
                cnt_prediction_tuple[(k,v)] += 1
    # Add answers with more than n/2 votes to the final prediction result
    prediction_voted = {}
    for (k,v) in cnt_prediction_tuple:
        if cnt_prediction_tuple[(k,v)] >= lowest_votes_for_O:
            prediction_voted[k] = v

    return prediction_voted

def compute_consistency_score(prediction_ls, voted_prediction=None):
    '''
    Vote for most consistent named entities from a set of predictions.
    Params:
        prediction_ls: list of prediction (dict);
        voted_prediction: voted prediction (dict).
    Returns:
        consistency_score: consist_score of voted prediction (dict)
    '''
    consistency_score_entities = {}
    # compute consistency score for voted answers
    if voted_prediction != None:
        for k, v in voted_prediction.items():
            consistency_score_entities[(k, v)] = 0
            for tmp_prediction in prediction_ls:
                if k in tmp_prediction and v==tmp_prediction[k]:
                    consistency_score_entities[(k, v)] += 1
    # compute consistency score for all answers
    else:
        for tmp_prediction in prediction_ls:
            for k,v in tmp_prediction.items():
                if (k,v) not in consistency_score_entities:
                    consistency_score_entities[(k,v)] = 0
                consistency_score_entities[(k,v)] += 1
    
    return consistency_score_entities

def combine_consistency_scores(consistency_scores, prediction_agg):
    consistency_score_agg = {}
    for tmp_consistency_score in consistency_scores:
        for k in tmp_consistency_score:
            mention, type = k[0], k[1]
            if mention in prediction_agg and type==prediction_agg[mention]:
                consistency_score_agg[k] = tmp_consistency_score[k]
    
    return consistency_score_agg

def collect_mention2labels(all_predictions: List[Dict]):
    mention2labels = {} # entity name: [entity label 1, entity label 2]
    for pred in all_predictions:
        mention = list(pred.keys())[0]
        type = pred[mention]
        if mention not in mention2labels:
            mention2labels[mention] = []
        mention2labels[mention].append(type)
    
    return mention2labels