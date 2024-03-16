dataname=wikigold

start_time="TIME_STAMP"

# generate prompts
python code/standard/GeneratePrompts.py \
        --dataname $dataname \
# asking LLMs
python code/standard/AskGPT.py \
        --dataname $dataname \
        --start_time $start_time
# compute evaluation results
python code/standard/ComputeMetric.py \
        --dataname $dataname \
        --start_time $start_time \

