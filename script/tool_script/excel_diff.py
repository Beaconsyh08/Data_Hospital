import pandas as pd
import glob
import os 

#把目录下所有重名的excel算diff
#excel_diff = excel_root_path_2 - excel_root_path_1 新版本减旧版本

excel_root_path_1 = "/root/data_hospital/0704v40/0519/"#旧版本
excel_root_path_2 = "/root/data_hospital/0704v40/0630/"#新版本
excel_diff_out_path = "/root/data_hospital/0704v40/diff/"#结果路径


excel_file_paths = glob.glob(os.path.join(excel_root_path_2,"*.xlsx"))

for excel_path_2 in excel_file_paths:
    excel_path_1 = excel_root_path_1+excel_path_2.split("/")[-1]
    excel_path_out =   excel_diff_out_path+excel_path_2.split("/")[-1].split(".")[0]+"_diff.xlsx"
    excel_pd_2 = pd.read_excel(excel_path_2,header=None)
    excel_pd_1 = pd.read_excel(excel_path_1,header=None)
    excel_dict_1 = {}
    excel_dict_2 = {}
    if(excel_pd_2.shape[1]==2):
        excel_pd_2 = excel_pd_2.dropna(axis=0,how='any')
        excel_pd_1 = excel_pd_1.dropna(axis=0,how='any')
        excel_pd_1[1] = excel_pd_1[1].astype("float")
        excel_pd_2[1] = excel_pd_2[1].astype("float")
        for inx,col in excel_pd_1.iterrows():
            cls_name = col[0]
            num = col[1]
            excel_dict_1[cls_name] = num

        for inx,col in excel_pd_2.iterrows():
            cls_name = col[0]
            num = col[1]
            excel_dict_2[cls_name] = num           

        for item in excel_dict_2:
            if item in excel_dict_1:
                excel_dict_2[item]-=excel_dict_1[item]

        result_excel = pd.DataFrame()
        result_excel["name"] = excel_dict_2.keys()
        result_excel["difference"] = excel_dict_2.values()
        result_excel.to_excel(excel_path_out)
        print("Excel saved to %s"% excel_path_out)        
    elif(excel_pd_2.shape[1]==3):
        excel_pd_2 = excel_pd_2.dropna(axis=0,how='any')
        excel_pd_1 = excel_pd_1.dropna(axis=0,how='any')
        excel_pd_1[1] = excel_pd_1[1].astype("float")
        excel_pd_2[1] = excel_pd_2[1].astype("float")
        excel_pd_1[2] = excel_pd_1[2].astype("float")
        excel_pd_2[2] = excel_pd_2[2].astype("float")
        for inx,col in excel_pd_1.iterrows():
            cls_name = col[0]
            num = col[1]
            rat = col[2]
            excel_dict_1[cls_name] = [num,rat]

        for inx,col in excel_pd_2.iterrows():
            cls_name = col[0]
            num = col[1]
            rat = col[2]
            excel_dict_2[cls_name] = [num,rat]            

        for item in excel_dict_2:
            if item in excel_dict_1:
                excel_dict_2[item]=[excel_dict_2[item][0]-excel_dict_1[item][0],excel_dict_2[item][1]-excel_dict_1[item][1]]

        result_excel = pd.DataFrame()
        result_excel["name"] = excel_dict_2.keys()
        result_excel["value_difference"] = [item[0] for item in excel_dict_2.values()]
        result_excel["ratio_difference"] = [item[1] for item in excel_dict_2.values()]
        result_excel.to_excel(excel_path_out)
        print("Excel saved to %s"% excel_path_out)