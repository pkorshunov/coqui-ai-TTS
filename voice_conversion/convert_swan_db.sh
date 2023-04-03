
# conda activate coqui-ai
# export PYTHONPATH=.
# ./voice_conversion/convert_swan_db.sh 
DFL_PYTHON="/idiap/temp/pkorshunov/miniconda3/envs/deepfacelab/bin/python"
DFL_SRC="/idiap/temp/pkorshunov/src/deepfakes/DeepFaceLab_Linux"

DB_DIR="/idiap/resource/database/SWAN-Idiap/IDIAP/session_01/iPad"

WORKSPACE="/idiap/temp/pkorshunov/dfswan"

# subjects_src=("00004" "00002" "00021" "00026" "00032" "00051" "00018" "00029" "00043" "00001" "00040" "00013" "00016" "00036" "00012" "00017" "00027" "00030" "00034")
# subjects_dst=("00014" "00015" "00022" "00024" "00033" "00052" "00050" "00047" "00058" "00006" "00049" "00029" "00035" "00042" "00013" "00020" "00031" "00034" "00003")
# subjects_src=("00004")
# subjects_dst=("00014")
subjects_src=("00014" "00015" "00022" "00024" "00033" "00052" "00050" "00047" "00058" "00006" "00049" "00029" "00035" "00042" "00013" "00020" "00031" "00034" "00003")
subjects_dst=("00004" "00002" "00021" "00026" "00032" "00051" "00018" "00029" "00043" "00001" "00040" "00013" "00016" "00036" "00012" "00017" "00027" "00030" "00034")

length=${#subjects_dst[@]}
for ((i=0;i<$length;i++)); do

    # target is who the generated WAV will sound like, which is a source in terms of deepfacelab
    target=$WORKSPACE/${subjects_src[$i]}/*${subjects_src[$i]}*_p_2.wav
    # source is who is going to be converted to target, which is a dst in terms of deepfacelab
    source=$WORKSPACE/${subjects_dst[$i]}/*${subjects_dst[$i]}*_p_2.wav
    # we convert all the files of the source into target
    ref=$WORKSPACE/${subjects_dst[$i]}/*${subjects_dst[$i]}*_*_2.wav
    python voice_conversion/yourtts_zs_vc_pair.py --target-files "${target}" --source-files "${source}" --reference-files "${ref}" --target-id ${subjects_src[$i]}

done