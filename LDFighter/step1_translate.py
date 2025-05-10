import sys
sys.path.append("/root/autodl-tmp/project/ldfighter/src/")
import pandas as pd
from translator.meta_seamless import MetaTranslator
import time
import json
from rqs.rq4_rq5.defense_adv_attack.load_adv_prompts import load_autodan_prompts,load_gcg_prompts

if __name__ == "__main__":
    ori_queries = load_autodan_prompts()
    save_path = "/root/autodl-tmp/project/ldfighter/exp_results/defense_adv/autodan/step1_multilingual_queries.json"
    trans = MetaTranslator()
    start = time.time()
    # # top-11 languages according to CI across four models
    langs = ['eng', 'fra', 'rus', 'spa', 'ces', 'swe', 'deu', 'dan', 'nob', 'nld', 'pol']
    langs.remove("eng")
    multilingual_data = {}
    for i, q in enumerate(ori_queries):
        print(f"===={i}=====")
        trans_list = {}
        for target_lng in langs:
            translated = trans.translate("eng", target_lng, q)
            trans_list[target_lng]= translated
        multilingual_data[q]=trans_list
    json_data = json.dumps(multilingual_data)
    # 保存为JSON文件
    with open(save_path, 'w') as json_file:
        json_file.write(json_data)
    print(time.time()-start)
        
        