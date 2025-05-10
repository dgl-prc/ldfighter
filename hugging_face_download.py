from huggingface_hub import login
from huggingface_hub import snapshot_download
from constant import CONFG
def download(repo_id):
    login(token=CONFG.HUGGING_FACE_TOKEN)
    snapshot_download(repo_id=repo_id)

if __name__ =="__main__":
    download("meta-llama/Llama-2-13b-chat-hf")