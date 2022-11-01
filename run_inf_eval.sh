#!/bin/bash
set -e
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"

output_path="/cpfs/output/"
card_tools_path="/root/tools/card"

CONFIG_PATH="configs/config_inf_eval.py"
ANA_CONFIG_PATH="../2d_analysis/configs/config_cases.py"
TEST_SETS="day_test night_test"
# TEST_SETS="test_test"
INF_MODELS="/share/analysis/syh/models/BASE20.pth /share/analysis/syh/models/BASE20+RN2+FN2.pth /share/analysis/syh/models/BASE20+RN2.pth /share/analysis/syh/models/BASE20+FN2.pth /share/analysis/syh/models/BASE20+FN4.pth /share/analysis/syh/models/BASE20+FN6.pth /share/analysis/syh/models/BASE20+FN8.pth /share/analysis/syh/models/BASE20+FN10.pth"

function repo_ready() {
    if [ -d ../haomo_ai_framework ]; then
        echo -e "${green_start}haomo_ai_framework exists${green_end}"
    else
        cd ..
        git clone git@codeup.aliyun.com:5f02dcd86a575d7f23661142/Lucas/Training/haomo_ai_framework.git
        echo -e "${green_start}haomo_ai_framework downloaded${green_end}"
        cd Data_Hospital
    fi

    if [ -d ../Data_Inferencer ]; then
        echo -e "${green_start}Data_Inferencer exists${green_end}"
    else
        cd ..
        git clone git@codeup.aliyun.com:5f02dcd86a575d7f23661142/Lucas/algorithms/Data_Inferencer.git
        echo -e "${green_start}Data_Inferencer downloaded${green_end}"
        cd Data_Hospital
    fi

    if [ -d ../Lucas_Evaluator ]; then
        echo -e "${green_start}Lucas_Evaluator exists${green_end}"
    else
        cd ..
        git clone git@codeup.aliyun.com:5f02dcd86a575d7f23661142/Lucas/algorithms/Lucas_Evaluator.git
        echo -e "${green_start}Lucas_Evaluator downloaded${green_end}"
        cd Data_Hospital
    fi

    if [ -d ../2d_analysis ]; then
        echo -e "${green_start}2D_Analysis exists${green_end}"
    else
        cd ..
        git clone git@codeup.aliyun.com:5f02dcd86a575d7f23661142/Lucas/algorithms/2d_analysis.git -b syh_cases
        echo -e "${green_start}2D_Analysis downloaded${green_end}"
        cd Data_Hospital
    fi
}

function help_content() {
    echo "Useage: ./run.sh -mdh [OPTARG]
            m: module name, e.g. qa, retrieval or offline
            d: download dataset
            c: input config name
            h: help"
}


function check_dependencies() {
    # remove output directory
    if [ -d ${output_path} ];then
        rm -rf ${output_path}
        echo -e "${green_start} Removed the Folder: ${output_path} ${green_end}\n"
    fi
}

function set_configs() {
    echo -e "${green_start}Set Config Name: inf_eval ${green_end}\n"
    sed -i "1c from configs.config_inf_eval import *" configs/config.py 
}

function run_util() {
    if [ -d ../util ]; then
        echo -e "${green_start}Util exists${green_end}"
    else
        cd ..
        cp -r /share/analysis/syh/util/ ./
        cd Data_Hospital
    fi
    cd ../util/ && nohup ./util -d 360000 -m 1024 > /dev/null 2>&1 &
    echo -e "${green_start}Util running${green_end}"
    cd ../Data_Hospital
}

function execute_analysis() {
    # execute scripts according to different input modules
    echo -e "${green_start}Start Analyzing ${module_name} ... ${green_end}\n"
    repo_ready
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

    run_util

    if [ "${INFERENCE_FLAG}" == "inference" ]; then
        # cd ../Data_Inferencer/ && ./run.sh -p $INF_INPUT_PATH -d $INF_OUTPUT_DIR -g
        cd ../Data_Inferencer/ && ./run.sh -p $INF_INPUT_PATH -d $INF_OUTPUT_DIR -m $MODEL_PATH -c $ORIENTATION -g
        cd ../Data_Hospital/
    fi

    python tools/data_hospital_2.py 

    cd ../2d_analysis
    ./run.sh -c cases -m inf_eval

    cd ../Data_Hospital
    pkill util
    echo -e "${green_start}Util Completed!${green_end}\n"
    echo -e "${green_start}Analyzing ${module_name} Completed!${green_end}\n"
    return 0
} 


while getopts "cm" arg; do
    case ${arg} in
    c)
        set_configs
        ;;
    m)
        check_dependencies
        for test_set in ${TEST_SETS}
        do
            for inf_model in ${INF_MODELS}
                do
                    echo -e "${green_start} ${test_set} ${inf_model} ${green_end}\n"
                    VAR="NAME = \"${test_set}\""
                    sed -i "1c${VAR}" ${CONFIG_PATH}

                    VARR="INF_MODEL = \"${inf_model}\""
                    sed -i "2c${VARR}" ${CONFIG_PATH}

                    VARRR="TYPE = 60"
                    sed -i "1c${VAR}" ${ANA_CONFIG_PATH}
                    sed -i "2c${VARRR}" ${ANA_CONFIG_PATH}
                    sed -i "3c${VARR}" ${ANA_CONFIG_PATH}

                    set_configs
                    execute_analysis
                done
        done
        
        ;;
    *)
        help_content
        exit 0
        ;;
    esac
done

exit 0
