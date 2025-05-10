import json
import random
import csv
from constant import CONFG,Dataset
from translator.meta_seamless import MetaTranslator
from utils.help_func import load_json

def load_data(dataType, size=-1,rndseed=202402):
        questions = [] 
        if dataType == CONFG.DATA_PATH.NQ:
            questions = load_nq2019()
        elif dataType == CONFG.DATA_PATH.NQ:
            questions = load_adv_bench()
        else:
            raise Exception("Unsupported dataset {}".format(dataType))

        if size != -1:
            random.seed(rndseed)
            return random.sample(questions,size)
        else:
            return questions

def load_nq2019():
    questions = []
    with open(CONFG.DATA_PATH.NQ, 'r') as f:
        # Parse JSON objects one by one
        for line in f.readlines():
            obj = json.loads(line)
            # Extract the question from each object
            questions.append(obj['question'])
    return questions

def load_adv_bench():
    questions = []
    with open(CONFG.DATA_PATH.ADV_BENCH, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        # Skip the header row
        next(csv_reader)
        # Iterate over each row in the CSV file
        for row in csv_reader:
           questions.append(row[0])
    return questions

def load_query_response(data_path):
    data = load_json(data_path)
    langs = MetaTranslator.get_mastered_language()
    qst_rep_dict = {}
    for question in data:
        rep_list = []
        for lang in langs:
            rep = question[lang]["Rep"] 
            rep_list.append(rep)
        qst_rep_dict[question["eng"]["Qst"]] = rep_list
    return qst_rep_dict

if __name__=="__main__":
    # q = load_data(Dataset.AdvBench)
    # print(len(q))
    # print(q[0])
    # print(len(["afr", "amh", "arb", "ary", "arz", "asm", "azj", "bel", "ben", "bos", "bul", "cat", "ceb", "ces", "ckb", "cmn", "cmn_Hant", "cym", "dan", "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz", "gle", "glg", "guj", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kat", "kaz", "khk", "khm", "kir", "kor", "lao", "lit", "lug", "luo", "lvs", "mai", "mal", "mar", "mkd", "mlt", "mni", "mya", "nld", "nno", "nob", "npi", "nya", "ory", "pan", "pbt", "pes", "pol", "por", "ron", "rus", "sat", "slk", "slv", "sna", "snd", "som", "spa", "srp", "swe", "swh", "tam", "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie", "yor", "yue", "zlm", "zul"]))
    rst = load_query_response("/root/autodl-tmp/project/ldfighter/expe_results/rq1_advbench_debug.json")
    for key in rst.keys():
        print(rst[key][0])
        break
            