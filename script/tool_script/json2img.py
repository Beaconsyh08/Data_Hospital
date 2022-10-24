import json
from tqdm import tqdm


inputfile = ""
outputfile = "/share/analysis/syh/white_list_qa.txt"


with open (inputfile) as input_file:
    input_list = [_.strip() for _ in input_file]
    
with open (outputfile, "w") as output_file:
    for json_path in tqdm(input_list, total=len(input_list), desc="Loading"):
        with open(json_path, 'r') as json_file:
            json_obj = json.load(json_file)
            img_url = json_obj["img_url"]
            
        output_file.writelines(img_url + "\n")