import json
import os
EXCLUDE_LANGS = ["amh","azj","eus","fuv","gaz","gle","ibo","kaz","khk","lug","luo","mni","mya","nya","ory","pan","sat","sna","snd","som","tam","uzn","yor","zul"]
langs = ["afr", "amh", "arb", "ary", "arz", "asm", "azj", "bel", "ben", "bos", "bul", "cat", "ceb", "ces", "ckb", "cmn", "cmn_Hant", "cym", "dan", "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz", "gle", "glg", "guj", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kat", "kaz", "khk", "khm", "kir", "kor", "lao", "lit", "lug", "luo", "lvs", "mai", "mal", "mar", "mkd", "mlt", "mni", "mya", "nld", "nno", "nob", "npi", "nya", "ory", "pan", "pbt", "pes", "pol", "por", "ron", "rus", "sat", "slk", "slv", "sna", "snd", "som", "spa", "srp", "swe", "swh", "tam", "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie", "yor", "yue", "zlm", "zul"]

def load_json(data_path):
    with open(data_path,"r") as f:
        data = json.load(f)
        return data

def save_as_json(data,data_path, indent):
    with open(data_path,"w") as f:
        json.dump(data,f,indent=indent)
        

if __name__ == "__main__":
    # data_pth = "/Users/gldong/Desktop/llm数据分析/ori_data/rq1_llama_vanilla_advbench.json"
    # save_pth = "/Users/gldong/Desktop/llm数据分析/llama2_vanilla_harmful"
    # data_pth = "/Users/gldong/Desktop/llm数据分析/ori_data/rq1_adv_gemma-7b.json_checkpoint_30"
    # save_pth = "/Users/gldong/Desktop/llm数据分析/gemma-7b_vanilla_harmful"
    # data_pth = "/Users/gldong/Desktop/llm数据分析/ori_data/vanilla_adv/rq1_adv_gemini-pro.json_checkpoint_30"
    # save_pth = "/Users/gldong/Desktop/llm数据分析/gemini-pro_vanilla_harmful"
    
    data_pth = "/Users/gldong/Desktop/llm数据分析/ori_data/vanilla_adv/rq1_adv_chatgpt.json_checkpoint_30"
    save_pth = "/Users/gldong/Desktop/llm数据分析/chatgpt_vanilla_harmful"
    if not os.path.exists(save_pth):
        os.makedirs(save_pth) 
    data = load_json(data_pth)
    data_size = 30
    for i, element in enumerate(data):
        rep_list  = []
        for lang in element.keys():
            if lang not in EXCLUDE_LANGS:
                query = element[lang]["Qst"]
                if "Rep_En" in  element[lang]:
                    rep = element[lang]["Rep_En"]
                else:
                    rep = element[lang]["Rep"]
                record = {"query":query,"lang":lang,"text":rep,"label":""}
                rep_list.append(record)
        save_as_json(rep_list,f"{save_pth}/{i}.json",4)
        if i == data_size:
            break
        