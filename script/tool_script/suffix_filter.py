path = "/data_path/waterloo_snow.txt"
save_path = "/data_path/waterloo_snow_filtered.txt"

with open (path) as input_file:
    json_paths = [_.strip() for _ in input_file]

with open (save_path, "w") as output_file:
    for json_path in json_paths:
        if json_path.split("/")[-1].split("_")[-1].split(".")[0] in ["0000000030", "0000000060", "0000000090"]:
            output_file.writelines(json_path + "\n")
    
