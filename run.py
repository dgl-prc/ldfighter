import sys
sys.path.append("/root/autodl-tmp/project/ldfighter/src")
import json
import os
import time
import argparse
from constant import EXCLUDE_LANGS
from models.llama2 import LLAMA2
from models.gemma7b import Gemma7B
from models.gemini_pro import GeminiPro
from models.chatgpt import ChatGPT
from constant import CONFG, LLMType
from utils.help_func import save_file,load_json
from datetime import datetime
from langdetect import detect,lang_detect_exception
from translator.meta_seamless import MetaTranslator

"""
This script is to get the initial results of multilingual response from various models over two dataset.
"""   

def getLLM(args):
    if args.llm == LLMType.Llama2_13B:
        print("Llama2_13B")
        return LLAMA2(temperature=0.0,max_seq_len=2048)
    elif args.llm == LLMType.Gemma_7B:
        print("Gemma_7B")
        return Gemma7B(temperature=0.0,max_gen_len=2048,device=args.device)
    elif args.llm == LLMType.ChatGPT:
         print("ChatGPT")
         return ChatGPT(engine_name="gpt3-5", max_try_times=5)
    elif args.llm == LLMType.Gemini_pro:
       return GeminiPro()
    else:
        raise Exception("Unsuppoerted LLM Type")
    
def markdown_to_plain_text_list(markdown_text):
    """Gemma7B alway output very formallyHighly formatted text which make the translation diffculty. To address this, we extract each line of the output and translate it line by line. 
    Args:
        markdown_text (_type_): _description_

    Returns:
        _type_: _description_
    """
    lines = markdown_text.split('\n')
    plain_text_list = []
    for line in lines:
        line = line.replace('*', '')
        line = line.rstrip('\n')
        if line.strip():
            plain_text_list.append(line.lstrip())    
    return plain_text_list

def make_record(rep_judge, llm, translator, query, rep, query_lang):
    if rep is None:
        return {"Qst":query,"Rep":"Unsupported supported language"}
    rep_lang = ""
    try:
        # only supprt a few languages. We only need to judege if the response is in English
        rep_lang = detect(rep)
    except lang_detect_exception.LangDetectException as e:
        rep_lang = "unknow"
    # translate reponse to english if it is not in english
    rep_eng = []
    if query_lang!= "eng" and rep_lang!="en":
        lines = markdown_to_plain_text_list(rep)
        for line in lines:
            rep_eng.append(translator.translate(query_lang, "eng", line))
        rep_eng = "\n".join(rep_eng)
        
    if rep_judge and rep_lang == "en":
        scores = llm.response_judge(query, rep)
        return {"Qst":query,"Rep":rep,"Scores":scores}
    else:
        # rep_lang == "en" or query_lang == "eng"
        if len(rep_eng) == 0:
            return {"Qst":query,"Rep":rep}
        else:
            return {"Qst":query,"Rep":rep,"Rep_En":rep_eng}
    
def query_and_score(llm, translator, data_path, rst_save_path, last_check_point, rep_judge, num_samples):
    loaded_data = []
    print("=====loading data=====")
    with open(data_path, 'r', encoding='utf-8') as json_file:
        loaded_data = json.load(json_file)
    query_log = [] if last_check_point == -1 else load_json(rst_save_path+f"_checkpoint_{last_check_point}")
    for i, query_eng in enumerate(loaded_data.keys()):
        if i <= last_check_point:
            continue
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"{current_timestamp}>>>>>>Progress: {i}/{len(loaded_data.keys())}<<<<<<<")
        record = {}
        trans_dict = loaded_data[query_eng]
        rep = llm.simple_query(query_eng)
        record["eng"] = make_record(rep_judge, llm, translator, query_eng, rep, "eng")
        print("eng:"+json.dumps(record["eng"], indent=4,ensure_ascii=False))
        for lang in trans_dict.keys():
            # not all the LLMs support all the 98 languaes, so we need to exclude some langs.
            if lang in EXCLUDE_LANGS:
                continue
            query = trans_dict[lang]
            rep = llm.simple_query(query)
            record[lang] = make_record(rep_judge, llm, translator, query, rep, lang)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {lang}: "+json.dumps(record[lang], indent=4,ensure_ascii=False))
            if CONFG.DEBUG and len(record) > 3:
                break
        query_log.append(record)
        # save checkpoint
        if i > 0 and i%10 == 0:
            checkpoint = json.dumps(query_log, indent=4, ensure_ascii=False)
            save_file(checkpoint, rst_save_path+f"_checkpoint_{i}") 
        if CONFG.DEBUG and i > 2:
            break
        if i >= num_samples:
                break
    rst = json.dumps(query_log, indent=4, ensure_ascii=False)
    save_file(rst, rst_save_path) 

def main(args):
    start = time.time()
    llm = getLLM(args)
    translator = MetaTranslator(args.device)
    if args.dataset == "nq":
        data_path = CONFG.MULTILINGUAL_DATA_PATH.NQ
    elif args.dataset == "adv":
        data_path = CONFG.MULTILINGUAL_DATA_PATH.ADV_BENCH
    else:
        print(f"Unsuppoerted dataset:{args.dataset}")
    save_path = os.path.join(CONFG.EXPER_RST_SAVE_PATH, f"rq1_{args.dataset}_{args.llm}.json")

    print(f"========Excercising on {args.dataset}==========")
    query_and_score(llm, translator, data_path, save_path, args.last_check_point,args.judge, args.num_samples)
    print(f"====Complete in Total {time.time()-start} seconds====")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["adv","nq"])
    parser.add_argument("--llm", type=str, required=True, choices=["llama2-13b", "gemma-7b", "chatgpt", "gemini-pro"])
    parser.add_argument("--num_samples",  type=int, help='Number of sample to use',required=True)
    parser.add_argument("--last_check_point",
                        type=int,
                        default=-1,)
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    # parser.add_argument("--judge", action='store_true')
    args = parser.parse_args()
    print(f"=========settings==========\n{args}")
    main(args)