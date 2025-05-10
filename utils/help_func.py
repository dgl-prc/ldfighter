import json

def save_file(data, save_path):
    with open(save_path, 'w') as f:
        f.write(data)
def load_json(data_path):
    with open(data_path,"r") as f:
        data = json.load(f)
        return data
       
def save_as_json(data,data_path, indent):
    with open(data_path,"w") as f:
        json.dump(data,f,indent=indent)
