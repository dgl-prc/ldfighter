import sys

sys.path.append("/root/autodl-tmp/project/ldfighter/src/")
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.llama2_7b import LLAMA2_7B
from rqs.rq4_rq5.defense_adv_attack.load_adv_prompts import (
    load_autodan_prompts,
    load_autodan_prompts_with_sys_prompt,
)
import re
import json


def valid_gcg_adv(llam2_7b):
    data_path = "/root/autodl-tmp/project/ldfighter/dataset/casper_adv_test_rst.csv"
    # data_path = "/root/autodl-tmp/project/ldfighter/dataset/llama2_7b_valided_gcg_adv_samples.csv"
    data = pd.read_csv(data_path)
    adv_inputs = data[data["jail_original"] == True]["query"].values.tolist()
    llam2_7b.use_template = False
    # adv_inputs  = data["Query"].values.tolist()
    print(len(adv_inputs))
    print("=================With ADV suffix!!=========================")
    cnt = 0
    i = 0
    jailbroken_resutls = []
    for input in adv_inputs:
        i += 1
        print(f"============={i}================")
        rst = llam2_7b.simple_query(input)
        if llam2_7b.attack_success(rst):
            cnt += 1
            jailbroken_resutls.append({"ID": i, "Query": input, "Rep": rst})
    print(f"Total success:{cnt}")
    pattern = re.compile(r"<</SYS>>(.*?)\[/INST\]", re.DOTALL)
    data = jailbroken_resutls
    adv_query_list = data["Query"].values.tolist()
    pure_adv_queries = []
    for q in adv_query_list:
        rst = re.search(pattern, q)
        if rst:
            pure_adv_queries.append(rst.group(1).strip())
    data["PureAdvQuery"] = pure_adv_queries
    print(len(pure_adv_queries), len(adv_query_list))
    data.to_csv("llama2_7b_valided_gcg_adv_samples.csv", index=False)


def valid_autodan_adv(llam2_7b):
    adv_inputs = load_autodan_prompts()
    # adv_inputs = load_autodan_prompts_with_sys_prompt()
    print(len(adv_inputs))
    success_ids = []
    for i, input in enumerate(adv_inputs):
        print(f"============={i}================")
        # if i == 0:
        #     print(input)
        rst = llam2_7b.simple_query(input)
        if llam2_7b.attack_success(rst):
            success_ids.append(i)
        else:
            print(json.dumps({"id": i, "input": input, "rep": rst}, indent=4))
    print(f"Total success:{len(success_ids)}/{len(adv_inputs)}")


if __name__ == "__main__":
    llam2_7b = LLAMA2_7B(
        device="cuda:0", jailbroken_tries=10, use_template=True, attack_test="autodan"
    )
    valid_autodan_adv(llam2_7b)
