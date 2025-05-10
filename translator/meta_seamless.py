import sys
sys.path.append("/root/autodl-tmp/project/ldfighter/src/")
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText
import torch
import json
from constant import *
class MetaTranslator:
    def __init__(self,device=None) -> None:
        self.processor = AutoProcessor.from_pretrained(CONFG.META_TRANSLATOR_MODEL_PATH)
        self.model = SeamlessM4Tv2ForTextToText.from_pretrained(CONFG.META_TRANSLATOR_MODEL_PATH)
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
    def translate(self, src_lang,tgt_lang,text):
        text_inputs = self.processor(text=text, src_lang=src_lang, return_tensors="pt")
        text_inputs.to(self.device)
        decoder_input_ids = self.model.generate(**text_inputs, tgt_lang=tgt_lang)[0].tolist()
        translated_text = self.processor.decode(decoder_input_ids, skip_special_tokens=True)
        return translated_text
    
    @staticmethod
    def get_mastered_language():
        return ["afr", "amh", "arb", "ary", "arz", "asm", "azj", "bel", "ben", "bos", "bul", "cat", "ceb", "ces", "ckb", "cmn", "cmn_Hant", "cym", "dan", "deu", "ell", "eng", "est", "eus", "fin", "fra", "fuv", "gaz", "gle", "glg", "guj", "heb", "hin", "hrv", "hun", "hye", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kat", "kaz", "khk", "khm", "kir", "kor", "lao", "lit", "lug", "luo", "lvs", "mai", "mal", "mar", "mkd", "mlt", "mni", "mya", "nld", "nno", "nob", "npi", "nya", "ory", "pan", "pbt", "pes", "pol", "por", "ron", "rus", "sat", "slk", "slv", "sna", "snd", "som", "spa", "srp", "swe", "swh", "tam", "tel", "tgk", "tgl", "tha", "tur", "ukr", "urd", "uzn", "vie", "yor", "yue", "zlm", "zul"]

    
if __name__=="__main__":
    import time
    from utils.load_data import load_data
    trans = MetaTranslator()
    start = time.time()
    questions = load_data(CONFG.DATA_PATH.NQ,size=520)
    langs = trans.get_mastered_language()
    langs.remove("eng")
    multilingual_data = {}
    for i, q in enumerate(questions):
        trans_list = {}
        for target_lng in langs:
            translated = trans.translate("eng", target_lng, q)
            trans_list[target_lng]= translated
        multilingual_data[q]=trans_list
    json_data = json.dumps(multilingual_data)
    # 保存为JSON文件
    with open(CONFG.MULTILINGUAL_DATA_PATH.NQ, 'w') as json_file:
        json_file.write(json_data)
    print(time.time()-start)