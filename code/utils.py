import logging, logging.config
import sys
import os
import json
import torch
from tqdm import tqdm
import pandas as pd
from decimal import Decimal

def copy_file_to_path(data, dir, path):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    if os.path.exists(path):
        raise ValueError(f"Path already exists: {path}")
    save_data(path, data)

def load_data(path, json_format=None):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        # each line is a json object
        if json_format=="lines":
            with open(path, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
        # whole dataset is a json object
        else:
            data = json.load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def save_data(path, data, json_format=None):
    if path.endswith(".txt"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data[:-1]:
                f.write(str(item) + "\n")
            f.write(str(data[-1]))
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            # each line is a json object
            if json_format=="lines":
                for item in data:
                    line = json.dumps(item, ensure_ascii=False)
                    f.write(line)
                    f.write('\n')
            # whole dataset is a json object
            else:
                f.write(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        raise ValueError(f"Wrong path for prompts saving: {path}")
    
def json2dict(json_list):
    d = dict()
    for item in json_list:
        k = list(item.keys())[0]
        v = item[k]
        d[k] = v
    return d

def dict2json(in_dict):
    out_json = []
    for k, v in in_dict.items():
        out_json.append(
            {k: v}
        )
    return out_json

def convert_format(data, target_format):
    if target_format == "dict":
        if isinstance(data, dict):
            data_new = data
        elif isinstance(data, list):
            data_new = json2dict(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    elif target_format == "json":
        if isinstance(data, list):
            data_new = data
        elif isinstance(data, dict):
            data_new = dict2json(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    else:
        raise NotImplementedError(f"convert_format() Not implemented for target_format={target_format}")

    return data_new

def format_json2str(in_json):
    out_str = "["
    for i_item, item in enumerate(in_json):
        k = list(item.keys())[0]
        v = item[k]
        out_str += "{\"%s\": \"%s\"}" % (k, v)        
        if i_item < len(in_json)-1:
            out_str += ", "
    out_str += "]"

    return out_str


def fetch_indexed_embs(train_embs, demo_data):
    '''
    train_embs: embs of whole training set.
    demo_data: demo set selected from training set.
    '''
    idxes = [x["idx"] for x in demo_data]
    idxes = torch.tensor(idxes).long()
    train_embs = torch.from_numpy(train_embs)
    demo_embs = torch.index_select(train_embs, dim=0, index=idxes)
    demo_embs = demo_embs.numpy()

    return demo_embs

def get_logger(file_name, log_dir, config_dir):
    config_dict = json.load(open( os.path.join(config_dir, 'log_config.json')))
    config_dict['handlers']['file_handler']['filename'] = os.path.join(log_dir,  file_name.replace('/', '-'))
    # config_dict['handlers']['file_handler']['level'] = "INFO"
    config_dict['root']['level'] = "INFO"
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger()

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


from typing import List, Dict

# count, rate, p/r/f1
def compute_metrics(args, data_responses:List[Dict], write_metric:bool=True):
    '''
    columns to be collected:
        ["Type", "Gold count", "Gold rate", "Pred count", "Pred rate", "Prec", "Rec", "F1"]
    '''
    
    id2label = args.id2label
    type2record = {}
    for lb in id2label:
        type2record[lb] = {"Type":lb, "Gold count":0, "Gold rate":0, "Pred count":0, "Pred rate":0, "Correct count":0, "Prec":0, "Rec":0, "F1":0}
    
    for i_item, item in enumerate(tqdm(data_responses, desc="compute metric")):
        # Calculate the accuracy of confidential annotations within pool size
        if hasattr(args, "confident") and args.confident and args.confident_sample_size:
            if i_item >= args.confident_sample_size:
                break

        curr_label = item["label"]
        if isinstance(curr_label, str):
            curr_label = eval(curr_label)
        if isinstance(curr_label, list):
            curr_label = json2dict(curr_label)
        curr_pred = item["prediction"]
        if isinstance(curr_pred, str):
            curr_pred = eval(curr_pred)
        if isinstance(curr_pred, list):
            curr_pred = json2dict(curr_pred)

        # remove empty pred
        if "" in curr_pred:
            del curr_pred[""]
        
        for tmp_mention, tmp_type in curr_label.items():
            type2record[tmp_type]["Gold count"] += 1
        
        ood_type_preds = []
        ood_mention_preds = []
        for tmp_mention, tmp_type in curr_pred.items():
            # ood type
            if tmp_type not in id2label:
                ood_type_preds.append({tmp_mention:tmp_type})
                continue
            type2record[tmp_type]["Pred count"] +=1
            # ood mention
            if tmp_mention not in item["sentence"]:
                ood_mention_preds.append({tmp_mention:tmp_type})
                continue
            # compare with gold
            if tmp_mention in curr_label and tmp_type == curr_label[tmp_mention]:
                type2record[tmp_type]["Correct count"] += 1

        # if len(ood_type_preds)>0:
        #     logger.info(f"OOD Type predictions:\n{ood_type_preds}")
        # if len(ood_mention_preds)>0:
        #     logger.info(f"OOD mention predictions:\n{ood_mention_preds}")
    
    # compute overall metrics
    n_gold_tot = sum([x["Gold count"] for x in type2record.values()])
    n_pred_tot = sum([x["Pred count"] for x in type2record.values()])
    n_correct_tot = sum([x["Correct count"] for x in type2record.values()])
    prec_tot = n_correct_tot / n_pred_tot if n_pred_tot else 0
    rec_tot = n_correct_tot / n_gold_tot if n_gold_tot else 0
    if prec_tot and rec_tot:
        f1_tot = 2*prec_tot*rec_tot / (prec_tot+rec_tot)
    else:
        f1_tot = 0
    # prec_tot = round(prec_tot,4)*100
    prec_tot = Decimal(prec_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
    # rec_tot = round(rec_tot,4)*100
    rec_tot = Decimal(rec_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
    # f1_tot = round(f1_tot,4)*100
    f1_tot = Decimal(f1_tot*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

    # compute classwise metrics
    for k in type2record:
        gold_count = type2record[k]["Gold count"]
        pred_count = type2record[k]["Pred count"]
        correct_count = type2record[k]["Correct count"]
        
        gold_rate = gold_count / n_gold_tot if n_gold_tot else 0
        pred_rate = pred_count / n_pred_tot if n_pred_tot else 0
        # gold_rate = round(gold_rate,4)*100
        gold_rate = Decimal(gold_rate*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # pred_rate = round(pred_rate,4)*100
        pred_rate = Decimal(pred_rate*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

        prec = correct_count / pred_count if pred_count else 0
        rec = correct_count / gold_count if gold_count else 0
        if prec and rec:
            f1 = 2*prec*rec / (prec+rec)
        else:
            f1 = 0
        # prec = round(prec,4)*100
        prec = Decimal(prec*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # rec = round(rec,4)*100
        rec = Decimal(rec*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")
        # f1 = round(f1,4)*100
        f1 = Decimal(f1*100).quantize(Decimal("0.01"),rounding="ROUND_HALF_UP")

        type2record[k]["Gold rate"] = gold_rate
        type2record[k]["Pred rate"] = pred_rate
        type2record[k]["Prec"] = prec
        type2record[k]["Rec"] = rec
        type2record[k]["F1"] = f1

    type2record["Total"] = {"Type":"ToTal", "Gold count":n_gold_tot, "Gold rate":100, "Pred count":n_pred_tot, "Pred rate":100, "Correct count":n_correct_tot, "Prec":prec_tot, "Rec":rec_tot, "F1":f1_tot}

    # convert to table format
    df_metrics = pd.DataFrame(list(type2record.values()))
    logger.info(f"===== Metrics =====\n{df_metrics}")
    if write_metric:
        metric_path = args.metric_path
        df_metrics.to_csv(metric_path, index=False)

import tiktoken
import logging
logger = logging.getLogger()
from const import my_openai_api_keys, model_list

def set_api_key(model_name, ports=None):
    model_publisher = model_list[model_name]["publisher"]
    if  model_publisher == "openai":
        api_keys = my_openai_api_keys
    else:
        assert ports != None
        api_keys = []
        for port in ports:
            api_keys.append({
                "key":"empty", 
                "set_base":True, 
                "api_base":"http://localhost:%s/v1" % port
            })
    
    return api_keys


def assert_gpt35_turbo_16k(args, chat_paradigm="standard"):
    if chat_paradigm == "standard":
        if args.few_shot_setting != "zs":
            if args.reason_hint is None and args.tool_aug is None:
                if args.few_shot_number >= 35:
                    assert args.model == "gpt-3.5-turbo-16k"
            else:
                if args.few_shot_number >= 10:
                    assert args.model == "gpt-3.5-turbo-16k"
    else:
        raise ValueError(f"Unrecognized chat_paradigm:{chat_paradigm}")


def max_tokens(model):
    n_max_tokens = model_list[model]["max_tokens"]
    return n_max_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        print("Warning: Not an official model of OpenAI. Set tokens_per_message, tokens_per_name = 0, 0")
        tokens_per_message, tokens_per_name = 0, 0

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


import time
import random
import openai
import logging
logger = logging.getLogger()

def run_llm(
        messages, 
        openai_key=None, 
        model_name="gpt-3.5-turbo", 
        temperature=0,
        stop=None,
        ports=None,
        return_text=True
):
    assert return_text == True
    agent = SafeOpenai(openai_key, model_name=model_name,ports=ports)

    rerequest_signal = "Give up and request again."

    if "text" in model_name or "instruct" in model_name:
        # Convert messages format to prompt format
        content_all = [x["content"] for x in messages]
        prompt = "\n".join(content_all)
        response = agent.text(
            model=model_name, 
            prompt=prompt, 
            temperature=temperature, 
            stop=stop
            )

        if return_text:
            response = response['choices'][0]['text']
        return response
    else:
        response = agent.chat(
            model=model_name, 
            messages=messages, 
            temperature=temperature, 
            stop=stop
            )
        
        if rerequest_signal in response:
            response = run_llm(
                messages=messages, 
                openai_key=openai_key, 
                model_name=model_name, 
                temperature=temperature,
                stop=stop,
                ports=ports,
                return_text=return_text
            )
        
        # if return_text:
        #     response = response['choices'][0]['message']['content']
        return response

class SafeOpenai:
    def __init__(self, keys=None, model_name=None, start_id=None, proxy=None, ports=None):

        if keys is None:
            raise "Please provide OpenAI Key."
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

        if start_id is None:
            start_id = random.sample(list(range(len(keys))), 1)[0]
            
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.keys)
        current_key = self.keys[self.key_id % len(self.keys)]
        openai.api_key = current_key["key"]
        if "org" in current_key:
            openai.organization = current_key["org"]
        if "set_base" in current_key and current_key["set_base"] is True:
            openai.api_base = current_key["api_base"]

    
    def set_next_api_key(self):
        self.key_id = (self.key_id + 1) % len(self.keys)
        current_key = self.keys[self.key_id]
        openai.api_key = current_key["key"]
        if "org" in current_key:
            openai.organization = current_key["org"]
        if "set_base" in current_key and current_key["set_base"] is True:
            openai.api_base = current_key["api_base"]
        
    def chat(self, 
            model, 
            messages, 
            temperature, 
            stop=None,
            sleep_seconds=2, 
            max_read_timeout=1, 
            max_http=1,
            DEBUG=False,
            stream_max_time=50000):
        timeout_cnt = 0
        http_cnt = 0
        while True:
            if timeout_cnt >= max_read_timeout:
                logger.info(f"timeout_cnt exceed max_read_timeout {max_read_timeout}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
            elif http_cnt >= max_http:
                logger.info(f"http_cnt exceed max_read_timeout {max_http}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
                                    
            completion = {'role': '', 'content': ''}
            try:
                start_time = time.time()
                response = openai.ChatCompletion.create(
                    model=model, 
                    messages=messages, 
                    temperature=temperature,
                    stop=stop,
                    timeout=60,
                    stream=True
                )
                
                for i_event, event in enumerate(response):
                    chunk_time = time.time() - start_time
                    if chunk_time > stream_max_time:
                        logger.info(f"Stream responded exceeds {stream_max_time}s. Give up and request again.")
                        return f"===Stream responded exceeds {stream_max_time}s. Give up and request again.==="
                    
                    if event['choices'][0]['finish_reason'] == 'stop':
                        if DEBUG:
                            pass
                            print(f'收到的完成数据: {completion}')
                        break
                    for delta_k, delta_v in event['choices'][0]['delta'].items():
                        if DEBUG:
                            print(f'流响应数据: {delta_k} = {delta_v}')
                        completion[delta_k] += delta_v
                # messages.append(completion)  # directly add msg into messages
                msg = completion['content']     # parse returned msg

                return msg
            
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                if "maximum context length" in str(e): # if input exceed max len, just return
                    logger.info(f"###### Exceed length, set response to empty str.")
                    return ""
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.APIConnectionError as e:
                logger.info(f"Failed to connect to OpenAI API: {e}")
                if "HTTPSConnectionPool" in str(e) and 'Remote end closed connection without response' in str(e):
                    http_cnt += 1
                    logger.info(f"http_cnt + 1 = {http_cnt}")                
                self.set_next_api_key()
                time.sleep(sleep_seconds)  
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}\n{self.keys[self.key_id]}")
                if "please check your plan and billing details" in str(e):
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1
                self.set_next_api_key()
                time.sleep(sleep_seconds)  
            except Exception as e:
                logger.info(str(e))
                # delete invalid API keys
                if "This key is associated with a deactivated account" in str(e):
                    print(f"deactivated error key: {self.keys[self.key_id]}")
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1

                    if self.key_id + 1 == len(self.keys):
                        logger.info("All keys are invalid!!!")
                        sys.exit()
                                                           
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "Read timed out" in str(e):
                    timeout_cnt += 1
                    logger.info(f"timeout_cnt + 1 = {timeout_cnt}")

                self.set_next_api_key()
                time.sleep(sleep_seconds)

        # return completion

    def text(self, 
            model, 
            prompt, 
            temperature, 
            stop=None,
            sleep_seconds=2, 
            max_read_timeout=1, 
            max_http=1):
        timeout_cnt = 0
        http_cnt = 0
        while True:
            if timeout_cnt >= max_read_timeout:
                logger.info(f"timeout_cnt exceed max_read_timeout {max_read_timeout}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."
            elif http_cnt >= max_http:
                logger.info(f"http_cnt exceed max_read_timeout {max_http}. Skip this request.")
                return f"Timeout exceeds {max_read_timeout}, skip."                            
            try:

                completion = openai.Completion.create(
                    model=model, 
                    prompt=prompt, 
                    temperature=temperature,
                    stop=stop,
                    timeout=3)
                break
            except openai.error.APIError as e:
                logger.info(f"OpenAI API returned an API Error: {e}")
                if "maximum context length" in str(e):
                    logger.info(f"###### Exceed length, set response to empty str.")
                    return ""
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.APIConnectionError as e:
                logger.info(f"Failed to connect to OpenAI API: {e}")
                if "HTTPSConnectionPool" in str(e) and 'Remote end closed connection without response' in str(e):
                    http_cnt += 1
                    logger.info(f"http_cnt + 1 = {http_cnt}")
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except openai.error.RateLimitError as e:
                logger.info(f"OpenAI API request exceeded rate limit: {e}\n{self.keys[self.key_id]}")
                if "please check your plan and billing details" in str(e):
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1
                self.set_next_api_key()
                time.sleep(sleep_seconds)
            except Exception as e:
                logger.info(str(e))
                if "This key is associated with a deactivated account" in str(e):
                    print(f"deactivated error key: {self.keys[self.key_id]}")
                    logger.info(f"delete key: {self.keys[self.key_id]}, key id: {self.key_id}")
                    del self.keys[self.key_id]
                    self.key_id -= 1

                    if self.key_id + 1 == len(self.keys):
                        logger.info("All keys are invalid!!!")
                        sys.exit()
                                                           
                if "This model's maximum context length is" in str(e):
                    logger.info('reduce_length')
                    return 'ERROR::reduce_length'
                if "Read timed out" in str(e):
                    timeout_cnt += 1
                    logger.info(f"timeout_cnt + 1 = {timeout_cnt}")

                self.set_next_api_key()
                time.sleep(sleep_seconds)
                
        # if return_text:
        #     completion = completion['choices'][0]['text']
        return completion
