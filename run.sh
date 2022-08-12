#!/bin/bash
set -e
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"

CURRENT_PATH=${PWD}

output_path="/cpfs/output/"

card_tools_path="/root/tools/card"
dataset_yaml=${CURRENT_PATH}"/dataset/dataset.yaml"

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

    sleep 1
    rm -rf /oss: /bucket-dataengine
    sleep 1
    mkdir /oss:
    sleep 1
    mkdir /bucket-dataengine
    sleep 1
    ln -s /data-engine/* /oss:
    sleep 1
    ln -s /data-engine/bucket-dataengine/* /bucket-dataengine

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
    echo -e "${green_start}start analyzing ${module_name} ... ${green_end}\n"
    if [ "${module_name}" == "qa" ]; then
        python setup.py install && python tools/analyze_qa.py

    elif [ "${module_name}" == "eda" ]; then
        python setup.py install && python tools/eda.py

    elif [ "${module_name}" == "data_hospital" ]; then
        python setup.py install && python tools/data_hospital.py        

    elif [ "${module_name}" == "data_hospital_2" ]; then
    # TODO
        # cd ../data_inference
        # ./run.sh -p /data_path/to_be_inf.txt
        python setup.py install && python tools/data_hospital_2.py 
    else
        echo -e "${red_start}invalid input module name !!! ${red_end}\n"
        help_content
    fi

    echo -e "${green_start}analyzing ${module_name} complete!${green_end}\n"
    return 0
}

while getopts "dc:m:h" arg; do
    case ${arg} in
    d)
        download_dataset
        ;;
    c)
        conifg_name=${OPTARG}
        set_configs
        ;;
    m)
        check_dependencies
        module_name=${OPTARG}
        execute_analysis
        ;;
    *)
        help_content
        exit 0
        ;;
    esac
done

exit 0

