import sys
sys.path.append("/root/autodl-tmp/project/ldfighter/src/")
import json
from models.llama2_7b import LLAMA2_7B
from translator.meta_seamless import MetaTranslator
from run import query_and_score

if __name__ == "__main__":
    llam2_7b = LLAMA2_7B(device="cuda:0", jailbroken_tries=10,use_template=True, attack_test='autodan') 
    translator = MetaTranslator()
    data_path = "/root/autodl-tmp/project/ldfighter/exp_results/defense_adv/autodan/step1_multilingual_queries.json"
    rst_save_path = "/root/autodl-tmp/project/ldfighter/exp_results/defense_adv/autodan/step2_multilingual_rep.json"
    query_and_score(llam2_7b, translator, data_path, rst_save_path, last_check_point=100,rep_judge=False,num_samples=249)
    
    
    