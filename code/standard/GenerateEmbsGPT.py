import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse
import openai
from openai.embeddings_utils import get_embedding
import random

# import package from other dir
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from const import my_openai_api_keys

def load_query_data(path):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def load_demo_data(path, demo_num):
    if demo_num:
        demo_data = json.load(open(path, "r", encoding="utf-8"))
    else:
        demo_data = list()

    return demo_data


def generate_emb_gpt(args, query_data):
    df = pd.DataFrame(columns=["sentence"])
    df["sentence"] = [x["sentence"] for x in query_data]

    openai.api_key = args.api_key["key"]
    if args.api_key["set_base"] is True:
        openai.api_base = args.api_key["api_base"]
    
    print(f"Obtaining embedding of {len(df)} samples.")
    print("Start time: {}".format(time.strftime("%m%d%H%M")))
    df["embedding"] = df.sentence.apply(lambda x: get_embedding(x, engine=args.emb_model))
    print("End time: {}".format(time.strftime("%m%d%H%M")))

    df["embedding"] = df.embedding.apply(np.array)
    
    return np.stack(df["embedding"])

def save_embs(path, embs):
    np.save(path, embs)

def main(args):
    # load data
    query_data = load_query_data(args.query_data_path)

    # generate and save embs
    embs = generate_emb_gpt(args, query_data)
    print(f"ems shape: {embs.shape}")
    save_embs(args.query_embs_path, embs)


def get_paths(args):
    # data loading path
    args.query_data_path = f"data/{args.dataname}/{args.datamode}.json"
    # emb saving path
    args.query_embs_path = f"data/{args.dataname}/{args.datamode}_GPTEmb.npy"
    
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--dataname", default="wikigold", type=str)
    parser.add_argument("--datamode", default="test", type=str, choices=["train", "test"])
    # model
    parser.add_argument("--emb_model", default="text-embedding-ada-002", type=str)
    parser.add_argument("--emb_encoding", default="cl100k_base", type=str)
    parser.add_argument("--max_token_len", default=8000, type=int)

    args = parser.parse_args()

    args = get_paths(args)

    args.api_key = random.choice(my_openai_api_keys)

    print("---------- Generate GPT emb ------------")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    main(args)

    

