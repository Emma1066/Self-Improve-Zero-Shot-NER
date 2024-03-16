dataname=wikigold
datamode=train_shuffle_42
demo_datamode=train_shuffle_42

few_shot_setting=zs

consistency=1
temperature=0.7
query_times=5

# SelectConfident.py
entity_level_selection="th_ent"

for entity_threshold in 3.0 4.0 5.0
do
first_level="entity"
# ComputeMetric.py
conf_select_method=${entity_level_selection}_${entity_threshold}

# Confident
confident_sample_size=0

# Response2Annotation.py
self_annotate_tag=std_c5
demo_setting=pool

include_emb=1

start_time="TIME_STAMP"

# SelectConfident
python code/self_consistent_annotation/confidence_selection/SC_all_ans_SelectConfident.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --temperature $temperature --query_times $query_times \
        --entity_level_selection $entity_level_selection --entity_threshold $entity_threshold \
        --first_level $first_level \
        --start_time $start_time

python code/self_consistent_annotation/ComputeMetric.py \
        --dataname $dataname \
        --datamode $datamode \
        --demo_datamode $datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --temperature $temperature --query_times $query_times \
        --start_time $start_time \
        --self_annotation \
        --confident --conf_select_method $conf_select_method \
        --confident_sample_size $confident_sample_size

# Response2Annotation
python code/self_consistent_annotation/confidence_selection/Response2Annotation.py \
        --dataname $dataname \
        --demo_datamode $demo_datamode \
        --few_shot_setting $few_shot_setting \
        --consistency $consistency --temperature $temperature --query_times $query_times \
        --conf_select_method $conf_select_method \
        --confident_sample_size $confident_sample_size \
        --self_annotate_tag $self_annotate_tag --demo_setting $demo_setting \
        --include_emb $include_emb \
        --start_time $start_time \

done