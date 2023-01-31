import json

json_path = "/data_path/snow.txt"


with open("/data_path/snow_img.txt", "w") as output_file:
    with open (json_path) as input_file:
        for file in input_file:
            with open(file.strip(), 'r') as f:
                json_obj = json.load(f)
                img_url = json_obj["imgUrl"]
                output_file.writelines(img_url + "\n")
            
            