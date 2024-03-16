import numpy as np
import json
import random
from tqdm import tqdm

dataname = "wikigold"
datamode = "train"
task = "NER"
seed = 42

shuffled_datamode = f"{datamode}_shuffle_{seed}"

# data loading path
train_data_path = f"OPENAI/data/{task}/{dataname}/{datamode}.json"
train_data_GPTEmb_path = f"OPENAI/data/{task}/{dataname}/{datamode}_GPTEmb.npy"

# data saving path
shuffled_train_data_path = f"OPENAI/data/{task}/{dataname}/{shuffled_datamode}.json"
shuffled_train_data_GPTEmb_path = f"OPENAI/data/{task}/{dataname}/{shuffled_datamode}_GPTEmb.npy"

train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
train_data_GPTEmb = np.load(train_data_GPTEmb_path)

idxes = list(range(len(train_data)))

random.seed(seed)
random.shuffle(idxes)

# Generate data files in the new order
shuffled_train_data = []
shuffled_train_data_GPTEmb = []
for idx in tqdm(idxes, desc="generate shuffled data"):
    shuffled_train_data.append(train_data[idx])
    shuffled_train_data_GPTEmb.append(train_data_GPTEmb[idx])
shuffled_train_data_GPTEmb = np.stack(shuffled_train_data_GPTEmb, axis=0)

# save new data file
with open(shuffled_train_data_path, "w", encoding="utf-8") as wf:
    wf.write(json.dumps(shuffled_train_data, indent=4, ensure_ascii=False))
np.save(shuffled_train_data_GPTEmb_path, shuffled_train_data_GPTEmb)
print(f"len(shuffled_train_data) = {len(shuffled_train_data)}")
print(f"shuffled_train_data_GPTEmb shape = {shuffled_train_data_GPTEmb.shape}")