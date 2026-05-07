# 可变参数列表
levels=("L3" "L4" "L5" "L6")      # 多个 level
sample_frames_list=(256 512)      # 多个 sample_frames
DEFAULT_SCALE_FACTOR=0.2          # 如果 DEFAULT_SCALE_FACTOR 也多值，可改为数组并加一层循环
model_name="qwen2.5-vl-72b"
caption_path="./captions"
video_path="/cache/tv_series_data"
num_workers=32

# Drama 到 questions 的映射（每个 drama 对应多个 question 文件）
declare -A DRAMA_QUESTIONS=(
    ["chen_mo_de_zhen_xiang"]="chen_mo_de_zhen_xiang_20260214_093400.json chen_mo_de_zhen_xiang_20260214_163000.json"
    ["huan_le_song"]="huan_le_song_20260214_093400.json huan_le_song_20260214_163000.json"
    ["shan_hai_qing"]="shan_hai_qing_20260214_093400.json shan_hai_qing_20260214_163000.json"
    ["da_qin_di_guo"]="da_qin_di_guo_zhi_zong_heng_20260214_093400.json da_qin_di_guo_zhi_zong_heng_20260214_163000.json"
    ["kuang_biao"]="kuang_biao_20260214_093400.json kuang_biao_20260214_163000.json"
    ["yi_qi_tong_guo_chuang_1"]="yi_qi_tong_guo_chuang_1_20260214_093400.json yi_qi_tong_guo_chuang_1_20260214_163000.json"
    ["ren_shi_jian"]="ren_shi_jian_20260227_160000.json ren_shi_jian_20260214_163000.json"
    ["zhan_chang_sha"]="zhan_chang_sha_20260214_093400.json zhan_chang_sha_20260214_163000.json"
    ["san_ti"]="san_ti_20260214_093400.json san_ti_20260214_163000.json"
    ["zhen_huan_zhuan"]="zhen_huan_zhuan_20260214_093400.json zhen_huan_zhuan_20260214_163000.json"
    ["friends"]="friends_20260215_205000.json friends_20260227_160000.json"
    ["downton_abbey"]="downton_abbey_20260215_205500.json downton_abbey_20260227_160000.json"
    ["lost"]="lost_20260227_160000.json"
)

# 四重循环：level → sample_frames → drama → question
for sample_frames in "${sample_frames_list[@]}"; do
    for level in "${levels[@]}"; do
        for drama in "${!DRAMA_QUESTIONS[@]}"; do
            if [ "$drama" == "kuang_biao" ]; then
                current_scale_factor=0.1
            else
                current_scale_factor=$DEFAULT_SCALE_FACTOR
            fi

            for questio_name in ${DRAMA_QUESTIONS[$drama]}; do

                image_path="./results/${level}/${sample_frames}/results.${current_scale_factor}/img_data/${drama}"
                question_path="./questions/${drama}/${questio_name}"

                echo "Running: level=${level}, frames=${sample_frames}, drama=${drama}, question=${questio_name}"

                python ./ask_vllm.py \
                        --level "${level}" \
                        --caption_path ${caption_path} \
                        --image_path ./results \
                        --drama "${drama}" \
                        --scale_factor "${current_scale_factor}" \
                        --sample_frames "${sample_frames}" \
                        --question_path "${question_path}" \
                        --video_path "${video_path}" \
                        --use_render_info \
                        --only_gen_img

                python ./ask_vllm.py \
                        --level "${level}" \
                        --caption_path "${caption_path}" \
                        --image_path "${image_path}" \
                        --drama "${drama}" \
                        --model_name "${model_name}" \
                        --question_path "${question_path}" \
                        --video_path "${video_path}" \
                        --num_workers "${num_workers}" \
                        --use_render_info \
                        --parallel
            done
        done
    done
done