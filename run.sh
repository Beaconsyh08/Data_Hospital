#!/bin/bash
set -e
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"


output_path="/cpfs/output/"

card_tools_path="/root/tools/card"

function help_content() {
    echo "Useage: ./run.sh -mdh [OPTARG]
            m: module name, e.g. qa, retrieval or offline
            d: download dataset
            c: input config name
            h: help"
}

function download_dataset() {
    echo -e "${green_start}Downloading Dataset ...${green_end}"

    cd ${card_tools_path} && ./run.sh -f ${dataset_yaml} && cd -

    echo -e "${green_start}Download Dataset Completed!${green_end}\n"

    # sleep 1
    # rm -rf /oss: /bucket-dataengine
    # sleep 1
    # mkdir /oss:
    # sleep 1
    # mkdir /bucket-dataengine
    # sleep 1
    # ln -s /data-engine/* /oss:
    # sleep 1
    # ln -s /data-engine/bucket-dataengine/* /bucket-dataengine

    echo -e "${green_start}Data Re-Link Completed!${green_end}\n"
}

function check_dependencies() {
    # remove output directory
    if [ -d ${output_path} ];then
        # rm -rf ${output_path}
        echo -e "${green_start} Removed the Folder: ${output_path} ${green_end}\n"
    fi
}

function set_configs() {
    echo -e "${green_start}Set Config Name: ${conifg_name} ${green_end}\n"
    sed -i "1c from configs.config_${conifg_name} import *" configs/config.py 
}

function execute_analysis() {
    # execute scripts according to different input modules
    echo -e "${green_start}Start Analyzing ${module_name} ... ${green_end}\n"
    python setup.py install 
    python tools/data_hospital.py
    IS_INFERENCE=$(python ./tools/data_hospital_passer.py)
    array=(`echo $IS_INFERENCE | tr ' ' ' '` )
    echo $array
    ORIENTATION=${array[-5]}
    MODEL_PATH=${array[-4]}
    INF_INPUT_PATH=${array[-3]}
    INF_OUTPUT_DIR=${array[-2]}
    INFERENCE_FLAG=${array[-1]}
    if [ "${INFERENCE_FLAG}" == "inference" ]; then
        # cd ../data_inferencer/ && ./run.sh -p $INF_INPUT_PATH -d $INF_OUTPUT_DIR -g
        cd ../data_inferencer/ && ./run.sh -p $INF_INPUT_PATH -d $INF_OUTPUT_DIR -m $MODEL_PATH -c $ORIENTATION -g
        cd ../data_hospital/
    fi

    python tools/data_hospital_2.py 

    echo -e "${green_start}Analyzing ${module_name} Completed!${green_end}\n"
    return 0
}

while getopts "c:dm" arg; do
    case ${arg} in
    c)
        conifg_name=${OPTARG}
        set_configs
        ;;
    d)
        dataset_y=${2:-dataset/dataset.yaml}
        dataset_yaml=${PWD}"/"${dataset_y}
        download_dataset
        ;;
    m)
        check_dependencies
        execute_analysis
        ;;
    *)
        help_content
        exit 0
        ;;
    esac
done



exit 0

