dataname=wikigold
datamode=train_shuffle_42

consistency=1
query_times=5
temperature=0.7

few_shot_setting=zs

annotation_size=500

start_time="TIME_STAMP"

# choices: majority_voting, two_stage_majority_voting
consistency_selection=two_stage_majority_voting
output_SC_all_answer=0
parse_response=0

# Self-annotating with two stage majority voting
python code/self_consistent_annotation/GeneratePrompts.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --self_annotation

python code/self_consistent_annotation/AskGPT.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --query_times $query_times --temperature $temperature \
        --self_annotation \
        --annotation_size $annotation_size \
        --start_time $start_time

python code/self_consistent_annotation/ComputeMetric.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --query_times $query_times --temperature $temperature \
        --consistency_selection $consistency_selection \
        --output_SC_all_answer $output_SC_all_answer \
        --start_time $start_time \
        --self_annotation \

# Obtain self-annotated demonstration set
demo_datamode=train_shuffle_42
confident_sample_size=0
self_annotate_tag=std_c5
demo_setting=pool
include_emb=1
python code/self_consistent_annotation/confidence_selection/Response2Annotation.py \
        --dataname $dataname \
        --demo_datamode $demo_datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --query_times $query_times --temperature $temperature \
        --confident_sample_size $confident_sample_size \
        --self_annotate_tag $self_annotate_tag --demo_setting $demo_setting \
        --include_emb $include_emb \
        --start_time $start_time \

