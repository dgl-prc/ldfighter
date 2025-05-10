import sys
sys.path.append("/root/autodl-tmp/ldfighter/src")
import json
from utils.help_func import load_json
import pandas as pd

def load_gcg_prompts():
    df = pd.read_csv("/root/autodl-tmp/project/ldfighter/dataset/llama2_7b_valided_gcg_adv_samples.csv")
    ori_queries = df["PureAdvQuery"].values.tolist()
    return ori_queries

def load_autodan_prompts():
    data = load_json("/root/autodl-tmp/AutoDAN/results/autodan_hga/llama2_0_normal.json")
    autodan_pure_prompts = []
    for id in data.keys():
        ele = data[id]
        is_success = ele["is_success"]
        if is_success:
            goal = ele["goal"]
            adv_template= ele["final_suffix"]
            user_prompt = adv_template.replace('[REPLACE]', goal.lower())
            autodan_pure_prompts.append(user_prompt)
    return autodan_pure_prompts



def delete_content_after_inst(input_string):
    # 查找 `[/INST]` 标记的位置
    index = input_string.find('[/INST]')
    
    # 如果找到了 `[/INST]` 标记，返回标记之前的内容；否则，返回原字符串
    if index != -1:
        return input_string[:index + len('[/INST]')]
    else:
        return input_string
    
def load_autodan_prompts_with_sys_prompt():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rqs.rq4_rq5.defense_adv_attack.attacks.AutoDAN.utils.string_utils import autodan_SuffixManager,load_conversation_template

    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/Llama-2-7b-chat-hf",
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    conv_template = load_conversation_template("llama2")
    data = load_json("/root/autodl-tmp/AutoDAN/results/autodan_hga/llama2_0_normal.json")
    autodan_pure_prompts = []
    for id in data.keys():
        ele = data[id]
        is_success = ele["is_success"]
        if is_success:
            goal = ele["goal"]
            adv_template= ele["final_suffix"]
            # user_prompt = adv_template.replace('[REPLACE]', goal.lower())
            suffix_manager = autodan_SuffixManager(tokenizer=tokenizer,
                                        conv_template=conv_template,
                                        instruction=goal,
                                        target=ele["target"],
                                        adv_string=adv_template)
            prompt = suffix_manager.get_prompt()
            autodan_pure_prompts.append(delete_content_after_inst(prompt))
    return autodan_pure_prompts



        
        
        
        
    
