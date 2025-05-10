from src.config import ReadConfigFiles

CONFG = ReadConfigFiles.cfg()

class Dataset:
    NQ= "NaturalQuestions"
    AdvBench = "AdvBench"
    
class LLMType:
    Llama2_13B = "llama2-13b"
    Gemma_7B = "gemma-7b"
    ChatGPT = "chatgpt"
    Gemini_pro = "gemini-pro"

LLM_JUDGE_FAILURE = "FAILURE"

EXCLUDE_LANGS = ["amh","azj","eus","fuv","gaz","gle","ibo","kaz","khk","lug","luo","mni","mya","nya","ory","pan","sat","sna","snd","som","tam","uzn","yor","zul"]

llms =  [LLMType.ChatGPT,LLMType.Gemini_pro,LLMType.Gemma_7B,LLMType.Llama2_13B]