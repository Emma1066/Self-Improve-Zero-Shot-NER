from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from utils import dict2json

class PromptPoolEnglish(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname
    
    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "You are an expert in named entity recognition. You are good at information extraction.\n"

        return task_hint

    def get_reason_hint(self, args):
        if args.reason_hint is None:
            return ""
        
        if args.reason_hint == "standard_cot":
            return ""
        
        if args.reason_hint == "rationale":
            if args.reason_hint_person == "first":
                reason_hint = "Let's infer named entities step by step from the text."
            else:
                reason_hint = "Please infer named entities step by step from the text."
            return reason_hint
        
        if args.reason_hint == "ToolUseTok":
            if args.reason_hint_person == "first":
                # reason_hint = "Let's infer named entities step by step from the given word segmentation."
                reason_hint = "Let's infer named entities step by step from the text based on the given word segmentation."
            else:
                # reason_hint = "Please infer named entities step by step from the given word segmentation."
                reason_hint = "Please infer named entities step by step from the text based on the given word segmentation."
            return reason_hint       
        
        if args.reason_hint == "ToolUsePos":
            if args.reason_hint_person == "first":
                # reason_hint = "Let's infer named entities step by step from the given Part-of-Speech tags."
                reason_hint = "Let's infer named entities step by step from the text based on the given Part-of-Speech tags."
            else:
                # reason_hint = "Please infer named entities step by step from the given Part-of-Speech tags."
                reason_hint = "Please infer named entities step by step from the text based on the given Part-of-Speech tags."
            return reason_hint   

        if args.reason_hint == "ToolUseCon":
            if args.reason_hint_person == "first":
                # reason_hint = "Let's infer named entities step by step from the given constituency tree."
                reason_hint = "Let's infer named entities step by step from the text based on the given constituency tree."
            else:
                # reason_hint = "Please infer named entities step by step from the given constituency tree."
                reason_hint = "Please infer named entities step by step from the text based on the given constituency tree."
            return reason_hint       

        if args.reason_hint == "ToolUseDep":
            if args.reason_hint_person == "first":
                # reason_hint = "Let's infer named entities step by step from the given dependency tree."
                reason_hint = "Let's infer named entities step by step from the text based on the given dependency tree."
            else:
                # reason_hint = "Please infer named entities step by step from the given dependency tree."
                reason_hint = "Please infer named entities step by step from the text based on the given dependency tree."
            return reason_hint                                  
                
        if args.reason_hint == "seg_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's segment the text into reasonable and semantically independent segments. Then, we recognize named entities based on the segmentation results. "
            else:
                reason_hint = "First, you should segment the text into reasonable and semantically independent segments. Then, you should recognize named entities based on the segmentation results. "
            return reason_hint
        
        if args.reason_hint == "noun_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's recognize the noun phrases. Then, we recognize named entities based on the noun phrases. "
            else:
                reason_hint = "First, you should recognize the noun phrases. Then, you should recognize named entities based on the noun phrases. "
            return reason_hint
        
        if args.reason_hint == "noun":
            if args.reason_hint_person == "first":
                reason_hint = "Let's first recognize the noun phrases, then recognize named entities based on the noun phrases. "
            else:
                reason_hint = "You should first recognize the noun phrases, then recognize named entities based on the noun phrases. "
            return reason_hint
        
        if args.reason_hint == "pos_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform Part-of-Speech tagging. Then, we recognize named entities based on the Part-of-Speech tags. "
            else:
                reason_hint = "First, you should perform Part-of-Speech tagging. Then, you should recognize named entities based on the Part-of-Speech tags. "
            return reason_hint    
        

        if args.reason_hint == "pos":
            if args.reason_hint_person == "first":
                reason_hint = "Let's first perform Part-of-Speech tagging, then recognize named entities based on the Part-of-Speech tags. "
            else:
                reason_hint = "You should first perform Part-of-Speech tagging, then recognize named entities based on the Part-of-Speech tags. "
            return reason_hint
        

        if args.reason_hint == "dep_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform dependency parsing. Then, we recognize named entities based on the dependency tree. "
            else:
                reason_hint = "First, you should perform dependency parsing. Then, you should recognize named entities based on the dependency tree. "
            return reason_hint

        if args.reason_hint == "con_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's perform constituency parsing. Then, we recognize named entities based on the constituency tree. "
            else:
                reason_hint = "First, you should perform constituency parsing. Then, you should recognize named entities based on the constituency tree. "
            return reason_hint          
        
        if args.reason_hint == "syn_conj":
            if args.reason_hint_person == "first":
                reason_hint = "First, let's understand the syntactic structure of the text. Then, we recognize named entities based on the syntactic structure. "
            else:
                reason_hint = "First, you should understand the syntactic structure of the text. Then, you should recognize named entities based on the syntactic structure. "
            return reason_hint            
         
        
    def get_task_instruction(self, args):
    
        # if args.dataname == "chemdner":
        #     task_instruct = "Please recognize the named entities labeled  \"Chemical\" in the text.Provide the answer in JSON format: [{\"Entity Name\": \"Entity Label\"}]. "
        #     return task_instruct    

        label_set = "Given entity label set: %s\n" % (args.id2label)
        
        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "Given the text and the corresponding word segmentation, please recognize the named entities in the given text. "
            if args.tool_aug == "ToolPos":
                given = "Given the text and the corresponding Part-of-Speech tags, please recognize the named entities in the given text. "
            if args.tool_aug == "ToolDep":
                given = "Given the text and the corresponding dependency tree, please recognize the named entities in the given text. "     
            if args.tool_aug == "ToolCon":
                given = "Given the text and the corresponding constituency tree, please recognize the named entities in the given text. "                              
        else:
            given = "Please recognize the named entities in the given text. "

        # ans_format = "Based on the given entity label set, provide answer in the JSON format: [{\"Entity Name\": \"Entity Label\"}]. "            
        ans_format = "Based on the given entity label set, provide answer in the following JSON format: [{\"Entity Name\": \"Entity Label\"}]. If there is no entity in the text, return the following empty list: []. "

        task_instruct = label_set + given + ans_format          

        # task_instruct = "Given entity label set: %s\nPlease recognize the named entities in the given text. Based on the entity label set, provide the answer in JSON format: [{\"Entity Name\": \"Entity Label\"}]. " % (args.id2label) 
            
        return task_instruct            


    def get_prompt_prefix(self, args):
        task_hint = self.get_domain_hint(args)
        task_instruction = self.get_task_instruction(args)
        # reason hint position: font or back
        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "f" else ""
        prefix = task_hint + task_instruction + reason_hint
        
        return prefix

    def get_prompt_for_demo(self, args, demo):
        demo_sent = demo["sentence"]
        # true label or predicted label
        if args.__contains__('self_annotate_tag') and args.self_annotate_tag:
            if args.self_annotate_tag == "gold":
                demo_label = demo["label"]
            else:
                if "_abl_" in args.self_annotate_tag:
                    eles = args.self_annotate_tag.split("_")
                    abl_method = eles[eles.index("abl")+1]
                    demo_label = demo["self_annotate"][f"prediction_abl_{abl_method}"]
                else:
                    demo_label = demo["self_annotate"]["prediction"]
        else:
            demo_label = demo["label"]

        if isinstance(demo_label, str):
            demo_label = eval(demo_label)  
        if isinstance(demo_label, dict):
            demo_label = dict2json(demo_label)

        demo_prompt = "\nText: %s" % (demo_sent)   

        if args.reason_hint is None and args.tool_aug is None:
            demo_prompt += "\nAnswer: %s" % demo_label
            return demo_prompt

        if not (args.reason_hint is None) and args.tool_aug is None:
            if args.reason_hint == "standard_cot":
                label_rationale = demo["label_rationale"]
                demo_prompt += "\nAnswer: \n%s" % (label_rationale)
                demo_prompt +=  "\nSo, the answer is:%s" % (demo_label)
                return demo_prompt
            
            if args.reason_hint == "rationale":
                if args.__contains__('self_annotate_tag') and args.self_annotate_tag != "gold":
                    demo_rationale = demo["self_annotate"][f"{args.self_annotate_tag}_rationale"]
                else:
                    demo_rationale = demo["label_rationale"]
                demo_prompt += "\nAnswer: \n%s" % (demo_rationale)
                demo_prompt +=  "\nIn summary, the recognized entities are:%s" % (demo_label)
                return demo_prompt

            if args.reason_hint == "key_noun":
                demo_key_nouns = demo["key_noun"]
                demo_prompt += "\nKey nouns: %s\nEntities: %s" % (demo_key_nouns, demo_label)           
                return demo_prompt
            if args.reason_hint == "key_noun_verb":
                demo_key_nouns = demo["key_noun_verb"]
                demo_prompt += "\nAnswer: \nKey nouns: %s\nKey verbs: %s\nEntities: %s" % (demo_key_nouns, demo_label)           
                return demo_prompt    
            
            if args.reason_hint in ["pos", "pos_conj"]:
                demo_pos = demo["tok_pos_pair_str"]
                # if isinstance(demo_pos, str):
                #     demo_pos = eval(demo_pos)
                # demo_pos = " ".join(demo_pos)
                demo_prompt += "\nAnswer: \nPart-of-Speech tagging: %s\nEntities: %s" % (demo_pos, demo_label)           
                return demo_prompt  
            
            if args.reason_hint in ["con", "con_conj"]:
                demo_con = demo["con_str"]
                demo_prompt += "\nAnswer: \nConstituency parsing: %s\nEntities%s" % (demo_con, demo_label)           
                return demo_prompt
            
            if args.reason_hint in ["dep", "dep_conj"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nAnswer: \Dependency parsing: %s\nEntities%s" % (demo_dep, demo_label)
                return demo_prompt

            if args.reason_hint == "pos_dep":
                demo_pos = demo["tok_pos_pair_str"]
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nAnswer: \nPart-of-Speech tagging: %s\nDependency parsing: %s\nEntities: %s" % (demo_pos, demo_dep, demo_label)           
                return demo_prompt          

            if args.reason_hint == "pos_con":
                demo_pos = demo["tok_pos_pair_str"]
                demo_con = demo["con_str"]
                demo_prompt += "\nAnswer: \nPart-of-Speech tagging: %s\nConstituency parsing: %s\nEntities: %s" % (demo_pos, demo_con, demo_label)           
                return demo_prompt     

        if args.reason_hint is None and not (args.tool_aug is None):
            
            if args.tool_aug in ["ToolPos"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\nPart-of-Speech tags: %s\nAnswer: %s" % (demo_pos, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolDep"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nDependency Tree: %s\nAnswer: %s" % (demo_dep, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolCon"]:
                demo_con = demo["con_str"]
                demo_prompt += "\nConstituency Tree: %s\nAnswer: %s" % (demo_con, demo_label)
                return demo_prompt                     
        
        if not (args.reason_hint is None) and not (args.tool_aug is None):
            if args.reason_hint in ["ToolUsePos"] and args.tool_aug=="ToolPos":
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\nPart-of-Speech tags: %s\nAnswer: %s" % (demo_pos, demo_label)        
                return demo_prompt
            
            if args.reason_hint in ["ToolUseCon"] and args.tool_aug=="ToolCon":
                demo_con = demo["con_str"]
                demo_prompt += "\nConstituency Tree: %s\nAnswer: %s" % (demo_con, demo_label)
                return demo_prompt    

            if args.reason_hint in ["ToolUseDep"] and args.tool_aug=="ToolDep":
                demo_dep = demo["trip_dep"]
                demo_prompt += "\nDependency Tree: %s\nAnswer: %s" % (demo_dep, demo_label)
                return demo_prompt              

        
    def get_prompt_postfix(self, args, query):
        sent = query["sentence"]

        if args.tool_aug:
            if args.tool_aug == "ToolTokCoarse":
                tok_coarse = query["tok/coarse"]   
                input_output_instruction = "\nText: %s\nTokenization: %s\nAnswer: " % (sent, tok_coarse)
            elif args.tool_aug == "ToolPos":
                pos = query["tok_pos_pair_str"]
                input_output_instruction = "\nText: %s\nPart-of-Speech tags: %s\nAnswer: " % (sent, pos)                    
            elif args.tool_aug == "ToolDep":
                dep = query["trip_dep"]
                input_output_instruction = "\nText: %s\nDependency tree: %s\nAnswer: " % (sent, dep)    
            elif args.tool_aug == "ToolCon":
                dep = query["con_str"]
                input_output_instruction = "\nText: %s\Consitituency tree: %s\nAnswer: " % (sent, dep)
            else:
                raise ValueError(f"Unrecognized tool_aug: {args.tool_aug}")                                        
        else:
                input_output_instruction = "\nText: %s\nAnswer: " % (sent)

        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "b" else ""
        postfix = input_output_instruction + reason_hint
        return postfix



class PromptPoolChinese(object):
    def __init__(self, dataname) -> None:
        self.dataname = dataname

    def get_domain_hint(self, args):
        if args.task_hint is None:
            return ""
        else:
            task_hint = "你是命名实体识别方面的专家。你很擅长信息抽取。\n"

        return task_hint

    def get_reason_hint(self, args):
        if args.reason_hint is None:
            return ""

        if args.reason_hint == "standard_cot":
            return ""
        
        if args.reason_hint == "rationale":
            if args.reason_hint_person == "first":
                reason_hint = "让我们从文本一步步推理出命名实体。"
            else:
                reason_hint = "请从文本一步步推理出命名实体。"
            return reason_hint

        if args.reason_hint == "ToolUseTok":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们从给定的分词结果一步步推理出命名实体。"
                reason_hint = "让我们基于给定的分词结果，从文本一步步推理出命名实体。"
            else:
                # reason_hint = "请从给定的分词结果一步步推理出命名实体。"
                reason_hint = "请基于给定的分词结果，从文本一步步推理出命名实体。"
            return reason_hint       
        
        if args.reason_hint == "ToolUsePos":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们从给定的词性标注一步步推理出命名实体。"
                reason_hint = "让我们基于给定的词性标注，从文本一步步推理出命名实体。"                
            else:
                # reason_hint = "请从给定的词性标注一步步推理出命名实体。"
                reason_hint = "请基于给定的词性标注，从文本一步步推理出命名实体。"                
            return reason_hint   

        if args.reason_hint == "ToolUseCon":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们从给定的成分树一步步推理出命名实体。"
                reason_hint = "让我们基于给定的成分树，从文本一步步推理出命名实体。"                 
            else:
                # reason_hint = "请从给定的成分树一步步推理出命名实体。"
                reason_hint = "请基于给定的成分树，从文本一步步推理出命名实体。"                   
            return reason_hint       

        if args.reason_hint == "ToolUseDep":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们从给定的依存树一步步推理出命名实体。"
                reason_hint = "让我们基于给定的依存树，从文本一步步推理出命名实体。"                 
            else:
                # reason_hint = "请从给定的依存树一步步推理出命名实体。"
                reason_hint = "请基于给定的依存树，从文本一步步推理出命名实体。"                          
            return reason_hint                                  
        
        if args.reason_hint in ["rationale0", "rationale0_label"]:
            if args.reason_hint_person == "first":
                reason_hint = "让我们从句法结构、词法的角度一步步推理出命名实体。"
            else:
                reason_hint = "你可以从句法结构、词法的角度一步步推理出命名实体。"
            return reason_hint
        
        if args.reason_hint in ["rationale0_giveup", "rationale0_label_giveup"]:
            if args.reason_hint_person == "first":
                reason_hint = "让我们从句法结构、词法的角度一步步推理出命名实体。如果没有合适的推理方法，可以放弃推理，直接识别实体。"
            else:
                reason_hint = "你可以从句法结构、词法的角度一步步推理出命名实体。如果没有合适的推理方法，可以放弃推理，直接识别实体。"
            return reason_hint        

        if args.reason_hint == "noun_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们识别名词。接着，我们基于名词识别命名实体。"
                # reason_hint = "首先，让我们识别名词。接着，我们基于识别的名词识别命名实体。"
            else:
                reason_hint = "首先，你应该识别名词。接着，你应该基于名词识别命名实体。"
                # reason_hint = "首先，你应该识别名词。接着，你应该基于识别的名词识别命名实体。"
            return reason_hint    
        
        if args.reason_hint == "pos_conj":
            if args.reason_hint_person == "first":
                # reason_hint = "首先，让我们进行词性标注。接着，我们基于词性标注识别命名实体。"
                reason_hint = "首先，让我们进行词性标注。接着，我们基于标注的词性识别命名实体。"
            else:
                # reason_hint = "首先，你应该进行词性标注。接着，你应该基于词性标注识别命名实体。"
                reason_hint = "首先，你应该进行词性标注。接着，你应该基于标注的词性识别命名实体。"
            return reason_hint
        
        if args.reason_hint == "con_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行成分句法解析。接着，我们基于成分树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行成分句法解析。接着，你应该基于成分树识别命名实体。"
            return reason_hint      

        if args.reason_hint == "dep_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们进行依存句法解析。接着，我们基于依存树识别命名实体。"
            else:
                reason_hint = "首先，你应该进行依存句法解析。接着，你应该基于依存树识别命名实体。"
            return reason_hint
        
        if args.reason_hint in ["tok_conj", "tok_fine_conj", "tok_coarse_conj"]:
            if args.reason_hint_person == "first":
                # reason_hint = "首先，让我们进行分词。接着，我们基于分词识别命名实体。"
                reason_hint = "首先，让我们进行分词。接着，我们基于分词结果识别命名实体。"
            else:
                # reason_hint = "首先，你应该进行分词。接着，你应该基于分词识别命名实体。"
                reason_hint = "首先，你应该进行分词。接着，你应该基于分词结果识别命名实体。"
            return reason_hint    

        if args.reason_hint == "syn_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们理解文本的句法结构。接着，我们基于句法结构识别命名实体。"
            else:
                reason_hint = "首先，你应该理解文本的句法结构。接着，你应该基于句法结构识别命名实体。"
            return reason_hint  
        
        if args.reason_hint == "key_phrase_conj":
            if args.reason_hint_person == "first":
                reason_hint = "首先，让我们识别文本中的关键短语。接着，我们基于关键短语识别命名实体。"
            else:
                reason_hint = "首先，你应该识别文本中的关键短语。接着，你应该基于关键短语识别命名实体。"
            return reason_hint        


        if args.reason_hint == "key_phrase":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先识别文本中的关键短语，再基于关键短语识别命名实体。"
            else:
                reason_hint = "你应该先识别文本中的关键短语，再基于关键短语识别命名实体。"
            return reason_hint

        
        if args.reason_hint == "key_noun":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先解析文本中的关键名词，再基于解析结果识别命名实体。"
            else:
                reason_hint = "你应该先解析文本中的关键名词，再基于解析结果识别命名实体。"
            return reason_hint
                

        if args.reason_hint == "key_noun_verb":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先解析文本中的关键名词和动词，再基于解析结果识别命名实体。"
            else:
                reason_hint = "你应该先解析文本中的关键名词和动词，再基于解析结果识别命名实体。"
            return reason_hint    

        if args.reason_hint == "key_noun_verb_con_dep":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如识别关键名词和关键动词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如识别关键名词和关键动词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            return reason_hint   
        
        if args.reason_hint == "key_noun_con_dep":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如识别关键名词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如识别关键名词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            return reason_hint   


        if args.reason_hint in ["noun_verb_con_dep", "parse_woa"]:
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如识别名词和动词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如识别名词和动词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            return reason_hint   
        
        if args.reason_hint == "noun_con_dep":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如识别名词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如识别名词、对文本进行成分解析和依存解析等，再基于文本解析的结果识别命名实体。"
            return reason_hint    
        
        if args.reason_hint == "noun":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先识别文本中的名词，再基于名词识别命名实体。"
                # reason_hint = "让我们先识别文本中的名词，再基于识别的名词识别命名实体。"
            else:
                reason_hint = "你应该先识别文本中的名词，再基于名词识别命名实体。"
                # reason_hint = "你应该先识别文本中的名词，再基于识别的名词识别命名实体。"
            return reason_hint        
        
        if args.reason_hint == "dep":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们先对文本进行解析，例如依存解析等，再基于解析的结果识别命名实体。"
                reason_hint = "让我们先对文本进行依存句法解析，再基于依存树识别命名实体。"
            else:
                # reason_hint = "你可以先对文本进行解析，例如依存解析等，再基于解析的结果识别命名实体。"
                reason_hint = "你可以先对文本进行依存句法解析，再基于依存树识别命名实体。"
            return reason_hint       

        if args.reason_hint == "con":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们先对文本进行成分句法解析，再基于解析的结果识别命名实体。"
                reason_hint = "让我们先对文本进行成分句法解析，再基于成分树识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行成分句法解析，再基于成分树识别命名实体。"
            return reason_hint   

        if args.reason_hint == "pos":
            if args.reason_hint_person == "first":
                # reason_hint = "让我们先对文本进行解析，例如词性标注等，再基于解析的结果识别命名实体。"
                reason_hint = "让我们先对文本进行词性标注，再基于标注的词性识别命名实体。"
            else:
                # reason_hint = "你可以先对文本进行解析，例如词性标注等，再基于解析的结果识别命名实体。"
                reason_hint = "你可以先对文本进行词性标注，再基于标注的词性识别命名实体。"
            return reason_hint     

        if args.reason_hint in ["tok_all", "tok_fine", "tok_coarse"]:
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行分词，再基于分词结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行分词，再基于分词结果识别命名实体。"
            return reason_hint      

        if args.reason_hint == "pos_key_noun":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如进行词性标注、识别关键名词等，再基于解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如进行词性标注、识别关键名词等，再基于解析的结果识别命名实体。"
            return reason_hint   
        
        if args.reason_hint == "key_noun_pos":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，包括识别关键名词、进行词性标注等，再基于解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，包括识别关键名词、进行词性标注等，再基于解析的结果识别命名实体。"
            return reason_hint       
        
        if args.reason_hint == "pos_dep":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如进行词性标注和依存解析等，再基于解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如进行词性标注和依存解析等，再基于解析的结果识别命名实体。"
            return reason_hint     

        if args.reason_hint == "pos_con":
            if args.reason_hint_person == "first":
                reason_hint = "让我们先对文本进行解析，例如进行词性标注和成分句法解析等，再基于解析的结果识别命名实体。"
            else:
                reason_hint = "你可以先对文本进行解析，例如进行词性标注和成分句法解析等，再基于解析的结果识别命名实体。"
            return reason_hint         

    def get_task_instruction(self, args):
        label_set = "给定实体标签集：%s\n" % (args.id2label)

        if args.tool_desc:
            assert args.tool_aug
            if args.tool_aug == "ToolTokCoarse":
                given = "给定文本和对应的分词结果，请识别文本中的命名实体。"
            elif args.tool_aug == "ToolPos":
                given = "给定文本和对应的词性标注，请识别文本中的命名实体。"
            elif args.tool_aug == "ToolDep":
                given = "给定文本和对应的依存树，请识别文本中的命名实体。"   
            elif args.tool_aug == "ToolCon":
                given = "给定文本和对应的成分树，请识别文本中的命名实体。"
            else:
                raise ValueError(f"Unrecognize tool_aug: {args.tool_aug}")                                
        else:
            given = "请识别给定文本中的命名实体。"

        # ans_format = "基于给定的实体标签集，以[{\"实体名称\": \"实体标签\"}]的json格式提供答案。"
        ans_format = "基于给定的实体标签集，以如下JSON格式提供答案：[{\"实体名称\": \"实体标签\"}]。如果文本中没有实体，请返回如下空列表：[]。"
            
        task_instruct = label_set + given + ans_format      

        # task_instruct = "给定实体标签集：%s\n请识别给定文本中的命名实体，并基于给定的实体标签集，以[{\"实体名称\": \"实体标签\"}]的json格式提供答案。" % (args.id2label)
        # task_instruct = "给定实体标签集：%s\n请基于给定的实体标签集识别给定文本中的命名实体，并以[{\"实体名称\": \"实体标签\"}]的json格式提供答案。" % (args.id2label)    

        return task_instruct    

    def get_prompt_prefix(self, args):
        task_hint = self.get_domain_hint(args)
        task_instruction = self.get_task_instruction(args)
        # reason hint位置放在font或back
        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "f" else ""
        prefix = task_hint + task_instruction + reason_hint
        
        return prefix


    def get_prompt_for_demo(self, args, demo):
        demo_sent = demo["sentence"]
        # true label or predicted label
        if args.__contains__('self_annotate_tag'):
            demo_label = demo["self_annotate"][args.self_annotate_tag]
        else:
            demo_label = demo["label"]
        if isinstance(demo_label, str):
            demo_label = eval(demo_label)  
        if isinstance(demo_label, dict):
            demo_label = dict2json(demo_label)

        demo_prompt = "\n文本：%s" % (demo_sent)

        if args.reason_hint is None and args.tool_aug is None:
            demo_prompt += "\n答案：%s" % demo_label
            return demo_prompt
        
        if not (args.reason_hint is None) and args.tool_aug is None:
            if args.reason_hint == "standard_cot":
                label_rationale = demo["label_rationale"]
                demo_prompt += "\答案：\n%s" % (label_rationale)
                demo_prompt +=  "\n所以，答案是：%s" % (demo_label)
                return demo_prompt
            
            if args.reason_hint == "rationale":
                if args.__contains__('self_annotate_tag') and args.self_annotate_tag != "gold":
                    demo_rationale = demo["self_annotate"][f"{args.self_annotate_tag}_rationale"]
                else:
                    demo_rationale = demo["label_rationale"]
                demo_prompt += "\n答案：\n%s" % (demo_rationale)
                demo_prompt +=  "\n综上，最终识别出的实体为：%s" % (demo_label)
                return demo_prompt
                        
            if args.reason_hint in ["rationale0", "rationale0_giveup"]:
                demo_rationale = demo["rationale_gpt4"]
                demo_prompt += "\n答案：%s" % (demo_rationale)           
                return demo_prompt 
            
            if args.reason_hint in ["rationale0_label", "rationale0_label_giveup"]:
                demo_rationale = demo["rationale_gpt4"]
                demo_prompt += "\n答案：\n推理：%s\n实体：%s" % (demo_rationale, demo_label)           
                return demo_prompt
            
            if args.reason_hint in ["key_phrase", "key_phrase_conj"]:
                demo_key_phrases = demo["key_phrases"]
                demo_prompt += "\n答案：\n关键短语：%s\n命名实体：%s" % (demo_key_phrases, demo_label)           
                return demo_prompt


            if args.reason_hint == "key_noun":
                demo_key_nouns = demo["key_noun"]
                demo_prompt += "\n答案：\n关键名词：%s\n命名实体：%s" % (demo_key_nouns, demo_label)           
                return demo_prompt
        
            
            if args.reason_hint in ["pos", "pos_conj"]:
                demo_pos = demo["tok_pos_pair_str"]
                # if isinstance(demo_pos, str):
                #     demo_pos = eval(demo_pos)
                # demo_pos = " ".join(demo_pos)
                demo_prompt += "\n答案：\n词性标注：%s\n命名实体：%s" % (demo_pos, demo_label)           
                return demo_prompt  
            
            if args.reason_hint in ["con", "con_conj"]:
                demo_con = demo["con_str"]
                demo_prompt += "\n答案：\n成分句法分析：%s\n命名实体：%s" % (demo_con, demo_label)           
                return demo_prompt    

            if args.reason_hint in ["dep", "dep_conj"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n答案：\n依存分析：%s\n命名实体：%s" % (demo_dep, demo_label)
                return demo_prompt              
            

            if args.reason_hint in ["tok_coarse", "tok_coarse_conj"]:
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n答案：\n分词：%s\n命名实体：%s" % (demo_tok_coarse, demo_label)           
                return demo_prompt

            if args.reason_hint == "tok_all":
                demo_tok_fine = demo["tok/fine"]
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n答案：\n细粒度分词：%s\n粗粒度分词：%s\n命名实体：%s" % (demo_tok_fine, demo_tok_coarse, demo_label)           
                return demo_prompt       


            if args.reason_hint == "pos_dep":
                demo_pos = demo["tok_pos_pair_str"]
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n答案：\n词性标注：%s\n依存解析：%s\n命名实体：%s" % (demo_pos, demo_dep, demo_label)           
                return demo_prompt          

            if args.reason_hint == "pos_con":
                demo_pos = demo["tok_pos_pair_str"]
                demo_con = demo["con_str"]
                demo_prompt += "\n答案：\n词性标注：%s\n成分句法解析：%s\n命名实体：%s" % (demo_pos, demo_con, demo_label)
                return demo_prompt
        
        if args.reason_hint is None and not (args.tool_aug is None):

            if args.tool_aug in ["ToolTokCoarse"]:
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n分词：%s\n答案：%s" % (demo_tok_coarse, demo_label)
                return demo_prompt
            
            if args.tool_aug in ["ToolPos"]:
                demo_pos = demo["tok_pos_pair_str"]
                demo_prompt += "\n词性标注：%s\n答案：%s" % (demo_pos, demo_label)
                return demo_prompt    

            if args.tool_aug in ["ToolDep"]:
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n依存树：%s\n答案：%s" % (demo_dep, demo_label)
                return demo_prompt

            if args.tool_aug in ["ToolCon"]:
                demo_con = demo["con_str"]
                demo_prompt += "\n成分树：%s\n答案：%s" % (demo_con, demo_label)
                return demo_prompt                     
        
        if not (args.reason_hint is None) and not (args.tool_aug is None):
            if args.reason_hint =="ToolUseTok" and args.tool_aug=="ToolTokCoarse":
                demo_tok_coarse = demo["tok/coarse"]
                demo_prompt += "\n分词：%s\n答案：%s" % (demo_tok_coarse, demo_label)         
                return demo_prompt
        
            if args.reason_hint =="ToolUsePos" and args.tool_aug=="ToolPos":
                demo_pos = demo["tok_pos_pair_str"]
                # if isinstance(demo_pos, str):
                #     demo_pos = eval(demo_pos)
                # demo_pos = " ".join(demo_pos)
                demo_prompt += "\n词性标注：%s\n答案：%s" % (demo_pos, demo_label)          
                return demo_prompt  

            if args.reason_hint == "ToolUseDep" and args.tool_aug=="ToolDep":
                demo_dep = demo["trip_dep"]
                demo_prompt += "\n依存树：%s\n答案：%s" % (demo_dep, demo_label)        
                return demo_prompt
            
            if args.reason_hint == "ToolUseCon" and args.tool_aug=="ToolCon":
                demo_con = demo["con_str"]
                demo_prompt += "\n成分树：%s\n答案：%s" % (demo_con, demo_label)         
                return demo_prompt
            
        
    def get_prompt_postfix(self, args, query):
        sent = query["sentence"]

        if args.tool_aug:
            if args.tool_aug == "ToolTokCoarse":
                tok_coarse = query["tok/coarse"]
                input_output_instruction = "\n文本：%s\n分词：%s\n答案：" % (sent, tok_coarse)
            elif args.tool_aug == "ToolPos":
                pos = query["tok_pos_pair_str"]  
                input_output_instruction = "\n文本：%s\n词性标注：%s\n答案：" % (sent, pos)
            elif args.tool_aug == "ToolDep":
                dep = query["trip_dep"]
                input_output_instruction = "\n文本：%s\n依存树：%s\n答案：" % (sent, dep)    
            elif args.tool_aug == "ToolCon":
                dep = query["con_str"]
                input_output_instruction = "\n文本：%s\n成分树：%s\n答案：" % (sent, dep)
            else:
                raise ValueError(f"Unrecognized tool_aug: {args.tool_aug}")
        else:
                input_output_instruction = "\n文本：%s\n答案：" % (sent)

        reason_hint = self.get_reason_hint(args) if args.reason_hint_pos == "b" else ""
        postfix = input_output_instruction + reason_hint
        return postfix

