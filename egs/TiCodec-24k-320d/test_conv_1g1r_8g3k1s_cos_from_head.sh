#!/bin/bash
source path.sh

# 1 baseline 1 group
ckpt_name=logs_convonly_Lib_1g1r_cos_from_head
# ckpt=logs_conv_only/logs_convonly_Lib_1g1r_cos_from_head/g_00315000_changed
ckpt=logs_conv_only/logs_convonly_Lib_1g1r_cos_from_head/g_00300000_changed
echo checkpoint path: ${ckpt}
output_root_dir=../Paper_Data/GEN
# the path of test wave
output=${output_root_dir}/${ckpt_name}

# dir_list=('../Paper_Data/GT/LibriTTS' '../Paper_Data/GT/VCTK' '../Paper_Data/GT/AISHELL-3' '../Paper_Data/GT/Musdb' '../Paper_Data/GT/Audioset' )
# outputdir=('/LibriTTS' '/VCTK' '/AISHELL-3' '/Musdb' '/Audioset' )  
dir_list=('../Paper_Data/GT/LibriTTS')
outputdir=('/LibriTTS')

for item in "${dir_list[@]}"
do
    echo "$item"
    echo "${output}${outputdir[$i]}"
    mkdir -p ${output}${outputdir[$i]}
    
    CUDA_VISIBLE_DEVICES=2 python3 ${BIN_DIR}/vqvae_copy_syn.py \
        --model_path=${ckpt} \
        --config_path=config_24k_320d_conv_1g1r_8g3k1s.json \
        --input_wavdir=$item \
        --outputdir=${output}${outputdir[$i]} \
        --num_gens=10000

    i=$((i+1))
done
