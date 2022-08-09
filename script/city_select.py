import pandas as pd
from script.tool_script.create_card_txt import card_generator


if __name__ == '__main__':
    DF_PATH = "/share/analysis/result/eda/dataframes/lucas_nabem.pkl"
    CITY_LST = ["Beijing", "Baoding", "UNK"]
    
    df = pd.read_pickle(DF_PATH)["df"]
    
    for city in CITY_LST:
        city_df = df[df.city == city]
        json_path_lst = list(set(city_df.json_path.to_list()))
        print(city, len(json_path_lst))
        
        file_path = "/data_path/%s_select.txt" % city
        with open(file_path, "w") as output_file:
            for json_path in json_path_lst:
                output_file.writelines(json_path + "\n")
        
        card_generator(100000, "syh_exp", "data_total", [file_path])
        
        
    
    
    
