import io
import gc
import re
import os
import cv2
import sys
import math
import time
import json
import openai
import base64
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from google import genai
import concurrent.futures
from functools import partial
from google.genai import types
from datetime import timedelta
from decord import VideoReader
from collections import defaultdict
from PIL import Image, ImageFont, ImageDraw
from openai import OpenAI, DefaultHttpxClient
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

BASE_URL = ""
API_KEY = ""

EPISODES_PER_SEASON = 365

MODEL_CKPT_MAP = {
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "qwen3-32b": "/cache/Qwen3-32B",
    "qwen2.5-vl-72b": "/cache/Qwen2.5-VL-72B-Instruct",
    "qwen2.5-vl-32b": "/cache/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-7b": "/cache/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-32b": "/cache/Qwen3-VL-32B-Instruct",
    "qwen3-vl-8b": "/cache/Qwen3-VL-8B-Instruct",
    "qwen3-vl-235b": "/cache/Qwen3-VL-235B-A22B-Instruct",
    "glm4.6v": "/cache/GLM-4.6V",
    "intern-vl-38b": "/cache/InternVL3_5-38B",
    "intern-vl-241b": "/cache/InternVL3_5-241B",
    "conan": "conan",
    "long_rl": "long_rl",
    "doubao": "",
    "qwen3-vl-32b-thinking": "/cache/Qwen3-VL-32B-Thinking", 
    "qwen3.5-27b": "Qwen3.5-27B"
}

replace_names ={'Robert Crawley Earl of Grantham':'Robert Crawley',
                'Cora Crawley Countess of Grantham':'Cora Crawley',
                'Violet Crawley  Dowager Countess of Grantham':'Violet Crawley',
                'Doctor at Moorfields Hospital':'Doctor at Moorfields Hospital',
                'Violet Crawley- Dowager Countess of Grantham':'Violet Crawley',
                'Cora Crawley- Countess of Grantham':'Cora Crawley',
                'Robert Crawley- Earl of Grantham':'Robert Crawley',
                'General Sir Herbert Strutt':'Herbert Strutt',
                'Violet Crawley Dowager Countess of Grantham':'Violet Crawley',
                'The Velvet Violin Club Guest':'The Velvet Violin Club Guest',
                'Hugh Shrimpie MacClare Marquess of Flintshire':'Hugh Shrimpie MacClare',
                'Susan MacClare Marchioness of Flintshire':'Susan MacClare',
                "Daisy Mason": "Daisy Robinson",
                "Baxter": "Phyllis Baxter",
                "Lady Sybil Branson": "Lady Sybil Crawley",
                "Gwen Harding": "Gwen Dawson",
                '其他':'Others'}


DRAMA_CAP_MAP = {
    'chen_mo_de_zhen_xiang': 'chen_mo_de_zhen_xiang_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20260216.json',     
    'huan_le_song': 'huan_le_song_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251028.json',  
    'shan_hai_qing': 'shan_hai_qing_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251101.json',
    'da_qin_di_guo': 'da_qin_di_guo_zhi_zong_heng_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',  
    'kuang_biao': 'kuang_biao_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251030.json',    
    'yi_qi_tong_guo_chuang_1': 'yi_qi_tong_guo_chuang_1_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',
    'downton_abbey': 'downton_abbey_hierarchical_caption_Hiera_0202_3API_briefadaptlength_20260216.json',            
    'ren_shi_jian': 'ren_shi_jian_hierarchical_caption_Hiera_1023_3API_briefadaptlength_20251028.json', 
    'zhan_chang_sha': 'zhan_chang_sha_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',
    'friends': 'friends_hierarchical_caption_Hiera_0205_3API_briefadaptlength_20260205.json',                  
    'san_ti': 'san_ti_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20260131.json',
    'zhen_huan_zhuan': 'zhen_huan_zhuan_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251028.json',
    'lost': 'lost_hierarchical_caption_Hiera_0205_3API_briefadaptlength_20260210.json'
}

DRAMA_NAME_MAP = {
    'chen_mo_de_zhen_xiang': ['chen_mo_de_zhen_xiang'],     
    'huan_le_song': ['huan_le_song'],  
    'shan_hai_qing': ['shan_hai_qing'],
    'da_qin_di_guo': ['da_qin_di_guo_zhi_zong_heng'],  
    'kuang_biao': ['kuang_biao'],    
    'yi_qi_tong_guo_chuang_1': ['yi_qi_tong_guo_chuang_1'],
    'downton_abbey': ['downton_abbey_1', 'downton_abbey_2','downton_abbey_3','downton_abbey_4','downton_abbey_5','downton_abbey_6'],            
    'ren_shi_jian': ['ren_shi_jian'], 
    'zhan_chang_sha': ['zhan_chang_sha'],
    'friends': ['friends_1', 'friends_2','friends_3','friends_4','friends_5','friends_6','friends_7','friends_8','friends_9','friends_10'],                  
    'san_ti': ['san_ti'],
    'zhen_huan_zhuan': ['zhen_huan_zhuan'],
    'lost': ['lost_1','lost_2','lost_3','lost_4','lost_5','lost_6']
}


MESSAGE_SYSTEM = """
### Task:
    You are an expert in understanding TV drama and film plots. Several screenshots have been uniformly extracted from a video clip, and these screenshots are provided in the order they appear in the video. There is now a multiple-choice question with options A, B, C, D, etc., based on this video segment. Your task is to select the correct answer according to the provided screenshots.
    Input Format (Example):
    Question: What color is Li Suhua's top when she sees Zhou Bingyi off at the train station?
    A. Red
    B. Green
    C. Blue
    D. Yellow
    Output Format:
    Please output your answer in the following format:
    [C]
    """

GROUP_MESSAGE_SYSTEM = """
### Task:
    You are an expert in understanding TV drama and film plots. Several screenshots have been uniformly extracted from a video clip and arranged into 2×2 composite images according to the following rules:
    Grouping Rule: Every four consecutive screenshots form one group, maintaining the original playback order of the video.
    Arrangement Rule: Within each group, the four screenshots are arranged into a 2-row by 2-column grid as follows:
    Row 1: Screenshot 1 (left), Screenshot 2 (right)
    Row 2: Screenshot 3 (left), Screenshot 4 (right)
    Position Requirement: The top-left position of the composite image corresponds to the 1st screenshot of the group, and the bottom-right position corresponds to the 4th screenshot.
    Overall Order: All generated 2×2 composite images are displayed in the same sequence as their appearance in the original video.
    Now, there is a multiple-choice question related to this video segment, with options labeled A, B, C, D, etc. Your task is to select the correct answer based on these screenshots.

    Input Format:
    Question: What color is Li Suhua's top when she sees Zhou Bingyi off at the train station?
    A. Red\nB. Green\nC. Blue\nD. Yellow\nE. Black

    Output Format (Example):
    Please output your answer in the following format:
    [C]
    """

prompt_template = """
    Question：{Question}
    {Options}
    """

class LLMInference:
    def __init__(self, args):
        self.drama = args.drama
        self.level = args.level
        self.sample_frames = args.sample_frames
        self.scale_factor = args.scale_factor
        self.image_path = args.image_path
        if self.drama not in self.image_path:
            self.image_path = os.path.join(self.image_path, 
                                        self.level, 
                                        str(self.sample_frames), 
                                        'results.{}'.format(self.scale_factor),
                                        'img_data',
                                        self.drama)
        if self.sample_frames != int(self.image_path.split('/')[-4]):
            self.sample_frames = int(self.image_path.split('/')[-4])

        self.short_model_name = args.model_name
        self.model_name = MODEL_CKPT_MAP[args.model_name]
        self.question_path = args.question_path
        assert self.drama in self.question_path
        assert '.json' in self.question_path
        self.video_paths = [os.path.join(args.video_path, drama_) for drama_ in DRAMA_NAME_MAP[self.drama]]
        
        self.render_subtitle = args.render_subtitle
        self.use_render_info = args.use_render_info
        self.parallel = args.parallel
        self.num_workers = args.num_workers
        resolution_dict = {
            'low': 'media_resolution_low',
            'medium': 'media_resolution_medium',
            'high': 'media_resolution_high',
            'ultra_high': 'media_resolution_ultra_high',
        }

        self.group_4x4 = args.group_4x4
        self.group_2x2 = args.group_2x2

        self.media_resolution = resolution_dict[args.media_resolution]
        self.caption_path = os.path.join(args.caption_path, DRAMA_CAP_MAP[self.drama])
        
        self.only_gen_img = args.only_gen_img

        self.level_info = {
            'L2': {
                'frames': [],
                'times': [],
            },
            'L3': {
                'frames': [],
                'times': [],
            },
            'L4': {
                'frames': [],
                'times': [],
            },
            'L5': {
                'frames': [],
                'times': [],
            },
            'L6': {
                'frames': [],
                'times': [],
            },
        }

        if ('qwen' in self.short_model_name or 'glm' in self.short_model_name
            or 'intern' in self.short_model_name):
            self.client = OpenAI(
                base_url= "http://localhost:22002/v1", 
                api_key="EMPTY")
        # elif 'gemini' in self.short_model_name:
        #     self.client = genai.Client(
        #         api_key=API_KEY,
        #         http_options={'api_version': 'v1alpha'})
        else:
            self.client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                http_client=DefaultHttpxClient(
                    verify=False,
                )
            )

        self.episode_info = {}
        if self.only_gen_img:
            cumulative_duration = 0.0
            for season, video_path in enumerate(self.video_paths):
                video_data = os.path.join(video_path, 'video_data')
                items = os.listdir(video_data)
                subdirs = [item for item in items if os.path.isdir(os.path.join(video_data, item))]
                subdirs.sort()
                video_list = os.listdir(os.path.join(video_data, subdirs[-1]))
                video_list.sort()
                for video_i in tqdm(video_list):
                    if '.mp4' in video_i or '.mkv' in video_i:
                        if 'fixed' in video_i:
                            continue
                        video_i_path = os.path.join(video_data, subdirs[-1], video_i)
                        vr = VideoReader(video_i_path)
                        fps = vr.get_avg_fps()
                        total_frames = len(vr)
                        duration = total_frames / fps
                        self.episode_info['{}_{}'.format(season+1, int(video_i[:-4]))] = {
                            'fps': fps,
                            'total_frames': total_frames,
                            'start_time': cumulative_duration,
                            'video_path': video_i_path,
                        }
                        cumulative_duration += duration
                        del vr
                        gc.collect()

            if len(self.episode_info) == 0:
                raise ValueError("No valid video found in the specified directory.")
            
    def sample_list_equally(self, lst, n):
        if n <= 0:
            return []
        if n >= len(lst):
            return lst[:]
        indices = np.linspace(0, len(lst) - 1, num=n, dtype=int)
        return [lst[i] for i in indices]
    
    def genai_inference(self, msg, top_p=0.01, temperature=0.1):
        retries = 5
        version = self.model_name
        parts = []
        msg_system = msg[0]['content']
        question = msg[1]['content'][0]['text']
        parts.append(types.Part(text=msg_system))
        parts.append(types.Part(text=question))
        for i in range(1, len(msg[1]['content'])):
            msg_type = msg[1]['content'][i]['type']
            msg_data = msg[1]['content'][i][msg_type]
            if msg_type == 'text':
                parts.append(types.Part(text=msg_data))
            else:
                parts.append(types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=msg_data['url'].split('base64,')[1],
                                    ),
                                media_resolution={"level": self.media_resolution}
                            )
                )
        
        for _ in range(retries):
            try:
                print(version,top_p,temperature)
                return self.client.models.generate_content(
                            model=version,
                            contents=[
                                    types.Content(
                                        parts=parts,
                                    )
                                ],
                        )
                
            except openai.RateLimitError:
                time.sleep(15)
                # embed()
            except openai.OpenAIError as e:
                print(f'ERROR: {e}')
                if "403" in str(e) and _ < 2:
                    print(f"Received 403 error. Retrying in ... seconds...")
                elif "403" in str(e) and _ == 2:
                    version='gpt-5'
                else:
                    # 非403错误或最后一次尝试失败
                    return None
            except Exception as e:
                print(f'ERROR: {e}')
                time.sleep(15)
    
    def openai_inference(self, msg, top_p=0.01, temperature=0.1):
        retries = 2
        version = self.model_name
        for _ in range(retries):
            try:
                current_sys_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(current_sys_time, self.drama, self.sample_frames, self.level, version, top_p, temperature)
                if _==4 and 'gemini-2.5' in version:
                    version=='gemini-2.0-flash-exp'
                    print('Change model from gemini-2.0-pro-exp to gemini-2.0-flash-exp')
                
                return self.client.chat.completions.create(
                    model=version,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                                "chat_template_kwargs": {"enable_thinking": False},},
                    messages=msg,
                    max_tokens=1000
                )
                
            except openai.RateLimitError:
                time.sleep(2)
                # embed()
            except openai.OpenAIError as e:
                print(f'ERROR: {e}')
                if "403" in str(e) and _ < 2:
                    print(f"Received 403 error. Retrying in ... seconds...")
                elif "403" in str(e) and _ == 2:
                    version='gpt-5'
                else:
                    # 非403错误或最后一次尝试失败
                    return None

        print(f"Failed after multiple retries.")
        return f"Unsuccessful: Failed after multiple retries."
    
    def update_qa_dict(self, dict_path):
        if os.path.isfile(dict_path):
            is_completed = True
            with open(dict_path, 'r', encoding='utf-8') as f:
                q_a_dict = json.load(f)
                qa_list = []
                q_id_new = int(list(q_a_dict.keys())[-1])
                for qa_k in q_a_dict.keys():
                    qa_list.append(q_a_dict[qa_k]['Q&A']['Question'])
                for level, q_info in self.questions.items():
                    for q_key, q_a_batch in q_info.items():
                        for q_id, q_a in q_a_batch['Q&A'].items():
                            if q_a['Question'] not in qa_list:
                                q_id_new += 1
                                q_a_dict[q_id_new] = {
                                    'time': q_a_batch['time'],
                                    'Q&A': q_a,
                                    'level': level,
                                    'clip_id': q_key,
                                    'q_id': q_id,
                                }
                for q_i, q_info in q_a_dict.items():
                    if 'result' not in q_info['Q&A'].keys():
                        is_completed = False
                        break
            return q_a_dict, is_completed
        else:
            return {}, False
        
    def build_dict(self):
        q_a_dict = {}
        q_id_new = -1
        for level, q_info in self.questions.items():
            for q_key, q_a_batch in q_info.items():
                for q_id, q_a in q_a_batch['Q&A'].items():
                    q_id_new += 1
                    q_a_dict[q_id_new] = {
                        'time': q_a_batch['time'],
                        'Q&A': q_a,
                        'level': level,
                        'clip_id': q_key,
                        'q_id': q_id,
                    }
        return q_a_dict

    def single_inference(self):
        result_path = os.path.dirname(os.path.dirname(self.image_path))
        qa_path = os.path.join(result_path, 'qa_data', self.drama, self.short_model_name)
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)

        dict_path = os.path.join(qa_path, 'q_a_dict.json')
        q_a_dict, is_completed = self.update_qa_dict(dict_path)

        if is_completed == True:
            return q_a_dict 
                
        if q_a_dict == {}:
            q_a_dict = self.build_dict()
            
        for key, q_info in q_a_dict.items():
            if 'result' in q_info['Q&A'].keys():
                continue
            time = q_info['time']
            time = self.get_level_time(time)
            if time is None:
                continue
            time_ = time.replace(':', '-')
            time_ = time_.replace('#', '-')
            if self.drama in ['kuang_biao', 'lost']:
                speaker_path = os.path.join(self.image_path, time_, 'spkeaker.json')
                with open(speaker_path, 'r', encoding='utf-8') as f:
                    speaker_info = json.load(f)

            image_file_path = os.path.join(self.image_path, time_, 'render' if self.use_render_info else 'raw')

            image_files = os.listdir(image_file_path)
            # image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
            image_files = sorted(image_files, key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
            base64_frames = []
            frame_names = []
            for image_file in image_files:
                jpg_path = os.path.join(image_file_path, image_file)
                img = Image.open(jpg_path)
                base64_frames.append(img)
                frame_names.append(image_file)

            if self.group_4x4:
                base64_frames, frame_names = self.get_frame_names_4x4(base64_frames, frame_names)
            elif self.group_2x2:
                base64_frames, frame_names = self.get_frame_names_2x2(base64_frames, frame_names)

            msg = [
                    {"role": "system", "content": MESSAGE_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_template.format(Question=q_info['Q&A']['Question'],
                                                                        Options=q_info['Q&A']['Options'])}
                    ]},
                ]
            
            if self.short_model_name in ['gpt-4.1', 'gpt-5']:
                #selected_indices = np.round(np.linspace(0, len(base64_frames) - 1, 48)).astype(int).tolist()
                selected_indices = range(len(base64_frames))
            else:
                selected_indices = range(len(base64_frames))

            for idx in selected_indices:
                frame = base64_frames[idx]
                buffered = io.BytesIO()
                frame.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                frame_base64 = base64.b64encode(image_bytes).decode('utf-8')
                msg[1]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
                if self.drama in ['kuang_biao', 'lost']:
                    text_info = ""
                    if isinstance(frame_names[idx], list):
                        for sub_idx, sub_frame_name in enumerate(frame_names[idx]):
                            speaker_key = sub_frame_name[:-4]
                            if speaker_info[speaker_key] != {}:
                                text_info += 'The {}th sub-image, '.format(sub_idx + 1) +  speaker_info[speaker_key]["text"] + ". "
                    else:
                        speaker_key = frame_names[idx][:-4]
                        if speaker_info[speaker_key] != {}:
                            text_info += speaker_info[speaker_key]["text"] + ". "
                            
                    if text_info != "":
                        msg[1]["content"].append(
                        {"type": "text", "text": "This is the subtitle information for the image above：{}".format(text_info)})

            result, total_tokens, cot = self.get_llm_response(msg, key)

            q_a_dict[key]['Q&A']['result'] = result
            q_a_dict[key]['Q&A']['total_tokens'] = total_tokens
            q_a_dict[key]['Q&A']['an4ques'] = q_info['Q&A']['Question']
            q_a_dict[key]['Q&A']['cot'] = cot

            with open(dict_path, "w", encoding="utf-8") as f:
                json.dump(q_a_dict, f, ensure_ascii=False, indent=4)

        return q_a_dict

    def get_frame_names_4x4(self, base64_frames, frame_names):
        # 用于存放生成的4x4大图的文件名
        output_dir = "output_4x4_grids"  # 拼接大图的保存路径
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # ================= 结果容器 =================
        frame_names_4x4 = []      # 存放生成的拼接大图的路径 (一维列表)
        frame_names_grouped = []  # 存放原始文件名 (二维嵌套列表: [[组1...], [组2...]])

        # ================= 常量定义 =================
        GROUP_SIZE = 16  # 每组图片数量
        GRID_ROWS = 4    # 行数
        GRID_COLS = 4    # 列数

        total_images = len(base64_frames)

        # ================= 循环处理 =================
        # 步长为16，遍历所有图片
        for i in range(0, total_images, GROUP_SIZE):
            # 1. 切片获取当前分组的数据
            batch_imgs = base64_frames[i : i + GROUP_SIZE]
            batch_names = frame_names[i : i + GROUP_SIZE]
            
            # 如果数据为空则停止
            if not batch_imgs:
                break

            # 2. 【核心需求】将当前组的文件名列表加入到大列表中
            # frame_names_grouped 结构变为: [['1.jpg'...'16.jpg'], ['17.jpg'...'32.jpg'], ...]
            frame_names_grouped.append(batch_names)

            # 3. 准备拼接画布
            # 获取单张小图尺寸 (假设所有图尺寸一致，取第一张为准)
            img_w, img_h = batch_imgs[0].size
            # 创建大图：宽=小图宽x4，高=小图高x4
            grid_img = Image.new('RGB', (img_w * GRID_COLS, img_h * GRID_ROWS), (255, 255, 255))

            # 4. 遍历当前组的16张图进行排版
            for idx, img in enumerate(batch_imgs):
                # 计算在4x4网格中的坐标
                # idx // 4 得到行号 (0, 0, 0, 0, 1, 1...)
                # idx % 4  得到列号 (0, 1, 2, 3, 0, 1...)
                row = idx // GRID_COLS
                col = idx % GRID_COLS
                
                x = col * img_w
                y = row * img_h
                
                grid_img.paste(img, (x, y))

            # 5. 保存拼接后的大图
            group_index = i // GROUP_SIZE
            save_name = f"grid_group_{group_index:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # grid_img.save(save_path)
            
            # 6. 记录大图路径
            frame_names_4x4.append(grid_img)
        return frame_names_4x4, frame_names_grouped

    def get_frame_names_2x2(self, base64_frames, frame_names):
        # 用于存放生成的2x2大图的文件名
        output_dir = "output_2x2_grids"  # 拼接大图的保存路径
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # ================= 结果容器 =================
        frame_names_2x2 = []      # 存放生成的拼接大图的路径 (一维列表)
        frame_names_grouped = []  # 存放原始文件名 (二维嵌套列表: [[组1...], [组2...]])

        # ================= 常量定义 =================
        GROUP_SIZE = 4  # 每组图片数量
        GRID_ROWS = 2    # 行数
        GRID_COLS = 2    # 列数

        total_images = len(base64_frames)

        # ================= 循环处理 =================
        # 步长为16，遍历所有图片
        for i in range(0, total_images, GROUP_SIZE):
            # 1. 切片获取当前分组的数据
            batch_imgs = base64_frames[i : i + GROUP_SIZE]
            batch_names = frame_names[i : i + GROUP_SIZE]
            
            # 如果数据为空则停止
            if not batch_imgs:
                break

            # 2. 【核心需求】将当前组的文件名列表加入到大列表中
            # frame_names_grouped 结构变为: [['1.jpg'...'16.jpg'], ['17.jpg'...'32.jpg'], ...]
            frame_names_grouped.append(batch_names)

            # 3. 准备拼接画布
            # 获取单张小图尺寸 (假设所有图尺寸一致，取第一张为准)
            img_w, img_h = batch_imgs[0].size
            # 创建大图：宽=小图宽x4，高=小图高x4
            grid_img = Image.new('RGB', (img_w * GRID_COLS, img_h * GRID_ROWS), (255, 255, 255))

            # 4. 遍历当前组的16张图进行排版
            for idx, img in enumerate(batch_imgs):
                # 计算在2x2网格中的坐标
                # idx // 2 得到行号 (0, 0, 0, 0, 1, 1...)
                # idx % 2  得到列号 (0, 1, 2, 3, 0, 1...)
                row = idx // GRID_COLS
                col = idx % GRID_COLS
                
                x = col * img_w
                y = row * img_h
                
                grid_img.paste(img, (x, y))

            # 5. 保存拼接后的大图
            group_index = i // GROUP_SIZE
            save_name = f"grid_group_{group_index:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # grid_img.save(save_path)
            
            # 6. 记录大图路径
            frame_names_2x2.append(grid_img)
        return frame_names_2x2, frame_names_grouped


    def parallel_inference(self):
        result_path = os.path.dirname(os.path.dirname(self.image_path))
        qa_path = os.path.join(result_path, 'qa_data', self.drama, self.short_model_name)
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)

        dict_path = os.path.join(qa_path, 'q_a_dict.json')
        q_a_dict, is_completed = self.update_qa_dict(dict_path)

        if is_completed == True:
            return q_a_dict
        
        # return q_a_dict
        
        if q_a_dict == {}:
            q_a_dict = self.build_dict()

        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:  # 根据需要调整max_workers的数量
            for key, q_info in q_a_dict.items():
                futures.append(executor.submit(self.process_item, key, q_info))

            for future in as_completed(futures):
                key, updates = future.result()
                if updates is not None:
                    q_a_dict[key]['Q&A'].update(updates)
                    with open(dict_path, "w", encoding="utf-8") as f:
                        json.dump(q_a_dict, f, ensure_ascii=False, indent=4)
        return q_a_dict

    def process_item(self, key, q_info):
        if 'result' in q_info['Q&A'].keys():
            return key, None  # 如果已经有结果，则直接返回

        time = q_info['time']
        time = self.get_level_time(time)
        if time is None:
            return key, None
        time_ = time.replace(':', '-')
        time_ = time_.replace('#', '-')
        if self.drama in ['kuang_biao', 'lost']:
            speaker_path = os.path.join(self.image_path, time_, 'spkeaker.json')
            with open(speaker_path, 'r', encoding='utf-8') as f:
                speaker_info = json.load(f)

        image_file_path = os.path.join(self.image_path, time_, 'render' if self.use_render_info else 'raw')
        image_files = os.listdir(image_file_path)
        image_files = sorted(image_files, key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
        base64_frames = []
        frame_names = []

        for image_file in image_files:
            jpg_path = os.path.join(image_file_path, image_file)
            img = Image.open(jpg_path)
            base64_frames.append(img)
            frame_names.append(image_file)

        if self.group_4x4:
            base64_frames, frame_names = self.get_frame_names_4x4(base64_frames, frame_names)
        elif self.group_2x2:
                base64_frames, frame_names = self.get_frame_names_2x2(base64_frames, frame_names)

        msg = [
            {"role": "system", "content": MESSAGE_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_template.format(Question=q_info['Q&A']['Question'], 
                                                                Options=q_info['Q&A']['Options'])}]},
        ]

        if self.short_model_name in ['gpt-4.1', 'gpt-5']:
            # selected_indices = np.round(np.linspace(0, len(base64_frames) - 1, 48)).astype(int).tolist()
            selected_indices = range(len(base64_frames))
        else:
            selected_indices = range(len(base64_frames))

        for idx in selected_indices:
            frame = base64_frames[idx]
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            frame_base64 = base64.b64encode(image_bytes).decode('utf-8')
            msg[1]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
            
            if self.drama in ['kuang_biao', 'lost']:
                text_info = ""
                if isinstance(frame_names[idx], list):
                    for sub_idx, sub_frame_name in enumerate(frame_names[idx]):
                        speaker_key = sub_frame_name[:-4]
                        if speaker_info[speaker_key] != {}:
                            text_info += 'The {}th sub-image, '.format(sub_idx + 1) +  speaker_info[speaker_key]["text"] + ". "
                else:
                    speaker_key = frame_names[idx][:-4]
                    if speaker_info[speaker_key] != {}:
                        text_info += speaker_info[speaker_key]["text"] + ". "
                        
                if text_info != "":
                    msg[1]["content"].append(
                    {"type": "text", "text": "This is the subtitle information for the image above: {}".format(text_info)})

        result, total_tokens, cot = self.get_llm_response(msg, key)
        return key, {'result': result, 'total_tokens': total_tokens, 'an4ques': q_info['Q&A']['Question'], 'cot': cot}
    
    def get_llm_response(self, msg, key, max_retries=2, delay_base=0.5):
        for attempt in range(1, max_retries + 1):
            # 确保获取到非 None 的 response
            response = None
            while response is None:
                # if'gemini' in self.short_model_name:
                #     response = self.genai_inference(msg)
                # else:
                response = self.openai_inference(msg)

            # if'gemini' in self.short_model_name:
            #     result = response.text
            #     total_tokens = response.usage_metadata.prompt_token_count
            # else:
            result = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
            stripped_result = result.strip()
            # match = re.match(r'^\[(.)\]$', stripped_result)
            match = re.search(r'\[([A-Za-z])\]', stripped_result) or \
                    re.search(r'<\|begin_of_box\|>([A-Za-z])<\|end_of_box\|>', stripped_result)
            if match:
                return match.group(1), total_tokens, stripped_result # 提取成功，立即返回

            # 提取失败，准备重试（最后一次不等待）
            if attempt < max_retries:
                wait_time = delay_base * (2 ** (attempt - 1))  # 指数退避
                print(f"[{key or 'Unknown'}] 第 {attempt} 次提取失败，响应: {repr(result)}，{wait_time:.2f}s 后重试...")
                time.sleep(wait_time)
            else:
                print(f"[{key or 'Unknown'}] 已达最大重试次数 ({max_retries})，仍无法提取有效格式。最终响应: {repr(result)}")

        return None, 0, None  # 所有重试均失败
    
    def read_caption_info(self):
        with open(self.caption_path, 'r', encoding='utf-8') as file:
            hierachical_data = json.load(file)

        L6_event = hierachical_data['L6_event']
        for L6_i in L6_event.keys():
            L6_i_info = L6_event[L6_i]

            L6_i_time = L6_i_info['L6_info']['L6_time']
            L6_i_frame = L6_i_info['L6_info']['L6_frame']
            self.level_info['L6']['frames'].append(L6_i_frame)
            self.level_info['L6']['times'].append(L6_i_time)

            L5_event = L6_i_info['sub_L5_event']
            for L5_i in L5_event.keys():
                L5_i_info = L5_event[L5_i]

                L5_i_time = L5_i_info['L5_info']['L5_time']
                L5_i_frame = L5_i_info['L5_info']['L5_frame']
                self.level_info['L5']['frames'].append(L5_i_frame)
                self.level_info['L5']['times'].append(L5_i_time)

                L4_event = L5_i_info['sub_L4_event']
                for L4_i in L4_event.keys():
                    L4_i_info = L4_event[L4_i]

                    L4_i_frame = L4_i_info['L4_info']['L4_frame']
                    L4_i_time = L4_i_info['L4_info']['L4_time']
                    self.level_info['L4']['frames'].append(L4_i_frame)
                    self.level_info['L4']['times'].append(L4_i_time)

                    L3_event = L4_i_info['sub_L3_event']
                    for L3_i in L3_event.keys():
                        L3_i_info = L3_event[L3_i]

                        L3_i_frame = L3_i_info['L3_info']['L3_frame']
                        L3_i_time = L3_i_info['L3_info']['L3_time']
                        self.level_info['L3']['frames'].append(L3_i_frame)
                        self.level_info['L3']['times'].append(L3_i_time)
                        
                        L2_event = L3_i_info['sub_L2_event']
                        for L2_i in L2_event.keys():
                            L2_i_info = L2_event[L2_i]

                            L2_i_frame = L2_i_info['L2_info']['L2_frame']
                            L2_i_time = L2_i_info['L2_info']['L2_time']
                            self.level_info['L2']['frames'].append(L2_i_frame)
                            self.level_info['L2']['times'].append(L2_i_time)

    def read_question_info(self):
        with open(self.question_path, 'r', encoding='utf-8') as f:
            self.questions = json.load(f)
        for q_level, q_info in self.questions.items():
            for q_id in q_info.keys():
                time = q_info[q_id]['time']
                time = self.get_level_time(time)
                if time is None:
                    continue
                time_ = time.replace(':', '-')
                time_ = time_.replace('#', '-')
                sampled_img_path = os.path.join(self.image_path, time_)
                render_path = os.path.join(sampled_img_path, 'render')

                # 1. 检查是否已经完成采样
                is_already_sampled = (
                    os.path.exists(render_path) and 
                    self.count_jpg_files(render_path) == self.sample_frames
                )

                sample_res = None # 初始化变量

                if not is_already_sampled:
                    current_sys_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('{}, drama: {}, level: {}, time period: {}, sample frames from video.....'.format(current_sys_time, self.drama, self.level, time))
                    sample_res = self.sample_frames_from_video(time, sampled_img_path)

                sampled_speaker_path = os.path.join(self.image_path, time_, 'spkeaker.json')
                if self.drama in ['kuang_biao', 'lost'] and not os.path.exists(sampled_speaker_path):
                    if is_already_sampled:
                        current_sys_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print('{}, drama: {}, level: {}, time period: {}, get speaker.....'.format(current_sys_time, self.drama, self.level, time))
                        sample_res = self.sample_frames_from_video(time, sampled_img_path, only_return_sample_res=True)
                    self.get_speaker(time, sample_res, has_season=True if self.drama in ['lost'] else False)

                if is_already_sampled:
                    continue

    def count_jpg_files(self, sampled_img_path):
        image_extensions = ('.jpg', '.jpeg', '.png')
        return sum(
            1 for f in os.listdir(sampled_img_path)
            if f.lower().endswith(image_extensions)
        )
    def epi_time_to_global(self, epi_time_str):
        """
        将 '0004-23:59:59.999' 或 'S1#0003-23:59:59.999' 转为全局秒数（float）
        """
        if '#' not in epi_time_str:
            # 兼容旧格式（无季信息），默认 season=1
            season = 1
            ep_and_time = epi_time_str
        else:
            season_str, ep_and_time = epi_time_str.split('#', 1)
            if not season_str.startswith('S'):
                raise ValueError(f"Invalid season format in '{epi_time_str}'")
            season = int(season_str[1:])  # S1 -> 1

        ep_str, time_str = ep_and_time.split('-', 1)
        episode = int(ep_str)
        h, m, s = time_str.split(':')
        local_sec = int(h) * 3600 + int(m) * 60 + float(s)

        total_episodes_before = (season - 1) * EPISODES_PER_SEASON + (episode - 1)
        return total_episodes_before * 86400.0 + local_sec
    
    def parse_time_range(self, time_range_str):
        """
        解析 '0003-23:59:00.000_0004-00:02:30.000' -> (start_global_sec, end_global_sec)
        """
        parts = time_range_str.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid time range format: '{time_range_str}'")
        
        start_part, end_part = parts
        start_sec = self.epi_time_to_global(start_part)
        end_sec = self.epi_time_to_global(end_part)
        
        # if start_sec > end_sec:
        #     raise ValueError(f"Invalid time range: start > end in '{time_range_str}'")
        
        return start_sec, end_sec

    def find_containing_source(self, source_times, target_time):
        """
        在 source_times 中找到完全包含 target_time 的那个元素。
        
        Args:
            source_times: List[str], each like '0001-..._0002-...'
            target_time: str, like '0004-..._0005-...'
        
        Returns:
            str: the matching source segment
        
        Raises:
            RuntimeError: if not found (shouldn't happen per problem statement)
        """
        tgt_start, tgt_end = self.parse_time_range(target_time)
        
        for seg in source_times:
            src_start, src_end = self.parse_time_range(seg)
            # Check full containment: [src_start, src_end] ⊇ [tgt_start, tgt_end]
            if src_start <= tgt_start + 10 and tgt_end <= src_end:
                return seg
        
        # raise RuntimeError(
        #     "No source segment contains the target time range. "
        #     "This contradicts the problem assumption."
        # )
    
    def get_level_time(self, time):
        if self.level == 'L2':
            assert time in self.level_info['L2']['times']
            return time
        level_i_times = self.level_info[self.level]['times']
        time = self.find_containing_source(level_i_times, time)
        return time
    
    def get_raw_imgs(self, video_path, episode_id_str, start_time=None, end_time=None):
        """
        直接接收视频路径进行处理
        """
        assert os.path.exists(video_path)

        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        if start_time is not None:
            start_sec = self.time_str_to_seconds(start_time)
            start_frame = math.floor(start_sec * fps)
            start_frame = max(0, start_frame)
        else:
            start_frame = 0

        if end_time is not None:
            end_sec = self.time_str_to_seconds(end_time)
            end_frame = math.ceil(end_sec * fps) - 1
            end_frame = min(total_frames - 1, end_frame)
        else:
            end_frame = total_frames - 1
        
        local_indices = []
        for frame_id in range(start_frame, end_frame + 1):
            # 返回格式: 季_集_帧号 (例如 5_0004_102)
            local_indices.append(
                f"{episode_id_str}_{frame_id}"
            )

        del vr
        gc.collect()  # 可选，强制回收

        return local_indices

    def _parse_time_str(self, time_part):
        """
        解析单个时间点字符串。
        格式: 'S5#0004-00:43:23.726' 或 '0004-00:43:23.726' (默认为None季)
        返回: (season_str, episode_int, time_str)
        """
        if '#' in time_part:
            season_str, rest = time_part.split('#')
            # 提取 'S5' 中的 5
            season = int(season_str.replace('S', '').replace('s', ''))
        else:
            rest = time_part
            season = 1 # 或者根据需求设为默认值

        episode_str, timestamp = rest.split('-')
        return season, int(episode_str), timestamp

    def sample_frames_from_video(self, time, save_path, only_return_sample_res=False):
        # 1. 解析开始和结束时间字符串
        # time 格式示例: 'S5#0004-00:43:23.726_S6#0001-00:05:10.123'
        t_start_part, t_end_part = time.split('_')
        
        # 解析出 (季, 集, 时间戳字符串)
        s_season, s_ep, s_time = self._parse_time_str(t_start_part)
        e_season, e_ep, e_time = self._parse_time_str(t_end_part)

        # 2. 将 episode_info 的 key 转换为有序列表
        # 字典是无序的，我们需要一个列表来确定 S5_0004 的下一集是 S5_0005 还是 S6_0001
        # 列表元素示例: ('5_4', 5, 4) -> (原始key, 季int, 集int)
        sorted_episodes = []
        for key in self.episode_info.keys():
            # 假设 key 是 '5_6' 格式
            parts = key.split('_')
            if len(parts) == 2:
                season_num = int(parts[0])
                ep_num = int(parts[1])
                sorted_episodes.append((key, season_num, ep_num))
        
        # 按季、集排序
        sorted_episodes.sort(key=lambda x: (x[1], x[2]))

        # 3. 定位开始和结束的索引
        start_idx = -1
        end_idx = -1

        # 构造目标 key 进行匹配 (注意：输入是 S5#0004，字典key是 '5_4'，需要去掉前导0)
        target_start_key = f"{s_season}_{s_ep}" 
        target_end_key = f"{e_season}_{e_ep}"

        for i, (key, season_n, ep_n) in enumerate(sorted_episodes):
            # 比较逻辑：直接比对 key，或者比对数字 (更稳健)
            if season_n == s_season and ep_n == s_ep:
                start_idx = i
            if season_n == e_season and ep_n == e_ep:
                end_idx = i

        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Error: Could not find start ({target_start_key}) or end ({target_end_key}) episode in episode_info.")
        if start_idx > end_idx:
            raise ValueError("Error: Start episode is after end episode.")

        frame_indices = []
        # 4. 遍历区间内的每一集
        for i in range(start_idx, end_idx + 1):
            key, season_n, ep_n = sorted_episodes[i]
            video_path = self.episode_info[key]['video_path']
            
            # 确定当前集的截取范围
            c_start_time = None
            c_end_time = None

            # 如果是第一集
            if i == start_idx:
                c_start_time = s_time
            
            # 如果是最后一集
            if i == end_idx:
                c_end_time = e_time
            
            # 生成唯一的 ID 前缀，例如 "5_0004"
            # 这样返回的帧ID就是 "5_0004_12345"，避免跨季帧号冲突
            episode_id_str = f"{season_n}_{str(ep_n).zfill(4)}"

            frame_indices += self.get_raw_imgs(
                video_path=video_path,
                episode_id_str=episode_id_str,
                start_time=c_start_time,
                end_time=c_end_time
            )

        save_raw_path = os.path.join(save_path, 'raw')
        os.makedirs(save_raw_path, exist_ok=True)

        frame_indices_indexes = np.linspace(0, len(frame_indices)-1, self.sample_frames)
        frame_indices_indexes = np.round(frame_indices_indexes).astype(int)
        frame_indices_indexes = np.clip(frame_indices_indexes, 0, len(frame_indices))
        frame_indices_indexes = frame_indices_indexes.tolist()

        selected_frame_indices = [frame_indices[frame_indices_index] for frame_indices_index in frame_indices_indexes]

        frames_by_episode = defaultdict(list)
        for selected_frame_indice in selected_frame_indices:
            parts = selected_frame_indice.split('_')
            season = parts[0]
            episode = int(parts[1])
            frame_idx = int(parts[2])
            
            sea_epi = '{}_{}'.format(season, episode)
            frames_by_episode[sea_epi].append(frame_idx)

        sample_res = []

        # 2. 遍历分组后的字典，每次只处理一个视频
        for sea_epi, frame_indices in frames_by_episode.items():
            # 获取视频路径
            video_path = self.episode_info[sea_epi]['video_path']
            
            # 初始化 VideoReader (仅针对当前这一个视频)
            vr = VideoReader(video_path)
            
            try:
                # 批量获取该视频下的所有目标帧
                # 注意：frame_indices 最好排序，某些 VideoReader 顺序读取效率更高
                frame_indices.sort() 
                images = vr.get_batch(frame_indices).asnumpy()
                
                # 将结果存入列表
                sample_res.append([sea_epi, images, frame_indices])
                
            except Exception as e:
                print(f"Error processing {sea_epi}: {e}")
                # 根据需求决定是否 continue 或 raise
            
            finally:
                # 关键步骤：处理完当前视频后，显式删除引用并回收内存
                del vr
                gc.collect()

        # vrs = {}
        # for selected_frame_indice in selected_frame_indices:
        #     season = selected_frame_indice.split('_')[0]
        #     episode = int(selected_frame_indice.split('_')[1])
        #     sea_epi = '{}_{}'.format(season, episode)
        #     if sea_epi in vrs.keys():
        #         continue
        #     video_path = self.episode_info[sea_epi]['video_path']
        #     vrs[sea_epi] = VideoReader(video_path)
        #     gc.collect()
            
        # sample_res = []
        # for vr_key in vrs.keys():
        #     sea_epi_frame_indices = []
        #     for selected_frame_indice in selected_frame_indices:
        #         season = selected_frame_indice.split('_')[0]
        #         episode = int(selected_frame_indice.split('_')[1])
        #         sea_epi = '{}_{}'.format(season, episode)
        #         if sea_epi == vr_key:
        #             sea_epi_frame_indices.append(int(selected_frame_indice.split('_')[2]))
        #     images = vrs[vr_key].get_batch(sea_epi_frame_indices).asnumpy()
        #     sample_res.append([vr_key, images, sea_epi_frame_indices])

        # del vrs
        # gc.collect()  # 可选，强制回收

        # for sample_item in sample_res:
        #     episode = sample_item[0]
        #     images = sample_item[1]
        #     frame_indices = sample_item[2]
        #     for im_id in range(images.shape[0]):
        #         img = Image.fromarray(images [im_id])
        #         if self.scale_factor != 1:
        #             img = self.resize_img(img)
        #         img.save(os.path.join(save_raw_path, 
        #                               str(episode).zfill(4) + '_' + str(frame_indices[im_id])+".jpg"))
        
        if only_return_sample_res:
            return sample_res
        self.render(sample_res, save_path)
        return sample_res

    def render(self, sample_res, save_path):
        def process_single_frame(im_id, images, frame_indices, frame_face_data_epi_sea, 
                             subtitle_data, save_render_path, season, episode, sea_epi):
            """
            处理单个视频帧的渲染工作
            
            Args:
                im_id (int): 当前帧的索引
                images (numpy.ndarray): 包含所有帧的图像数组
                frame_indices (list): 每个帧对应的帧ID列表
                frame_face_data_epi_sea (dict): 存储每帧人脸数据的字典
                subtitle_data (dict): 存储字幕数据的字典
                save_render_path (str): 渲染后图像的保存路径
                season (str): 季节信息
                episode (str): 集数信息
                sea_epi (str): 季集组合标识符
                
            Returns:
                None: 此函数不返回值，直接将处理后的图像保存到指定路径
            """
            frame = images[im_id]
            if self.scale_factor != 1:
                frame = self.resize_img_pil(frame)
            
            frame_id = frame_indices[im_id]
            
            # Draw Face Boxes
            if frame_id in frame_face_data_epi_sea.keys():
                face_data = frame_face_data_epi_sea[frame_id]
                frame = self.draw_box(face_data, frame, 0.43, sea_epi, self.scale_factor)

            # Render Subtitles
            if self.render_subtitle == True:
                frame = self.render_subtitles(frame, subtitle_data, frame_id)

            # Save Image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Determine filename based on path structure
            if 'S' in save_render_path.split('/')[-2]:
                filename = f"{season}_{episode}_{frame_id}.jpg"
            else:
                filename = f"{episode}_{frame_id}.jpg"
                
            full_save_path = os.path.join(save_render_path, filename)
            cv2.imwrite(full_save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

        for sample_item in sample_res:
            sea_epi = sample_item[0]
            images = sample_item[1]
            frame_indices = sample_item[2]

            frame_face_data_epi_sea = {}
            face_path = os.path.join(self.episode_info[sea_epi]['video_path'].split('/video_data')[0], 'face_data')

            items = os.listdir(face_path)
            for item in items:
                if 'insightface' in item:
                    break
            face_path = os.path.join(face_path, item)
            season = sea_epi.split('_')[0]
            episode = sea_epi.split('_')[1].zfill(4)
            face_path = os.path.join(face_path, episode+'.json')
            with open(face_path, 'r', encoding='utf-8') as fr:
                face_data = json.load(fr)
                for obj_idx, obj_info in face_data['object_list'].items():
                    assert obj_info['start_frame'] == obj_info['end_frame']
                    if obj_info['start_frame'] in frame_face_data_epi_sea.keys():
                        frame_face_data_epi_sea[obj_info['start_frame']].append(obj_info)
                    else:
                        frame_face_data_epi_sea[obj_info['start_frame']] = [obj_info]

            subtitle_data = None
            if self.render_subtitle == True:
                subtitle_path = os.path.join(self.episode_info[sea_epi]['video_path'].split('/video_data')[0], 'subtitle_data')
                with open(os.path.join(subtitle_path, 'version.json'), 'r', encoding='utf-8') as fr:
                    version_data = json.load(fr)
                    subtitle_file = version_data['subtitle_checking_step']['active_version']
                    subtitle_file_path = os.path.join(subtitle_path, subtitle_file, episode+'.json')
                    with open(subtitle_file_path, 'r', encoding='utf-8') as file:
                        subtitle_data = json.load(file)

            save_render_path = os.path.join(save_path, 'render')
            os.makedirs(save_render_path, exist_ok=True)
            # --- Multi-threading Implementation ---
            # We use max_workers to limit threads (e.g., os.cpu_count() or a fixed number like 8 or 16)
            # Adjust max_workers based on your disk I/O speed and CPU cores.
            with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
                # Create a partial function with fixed arguments that don't change per frame
                worker = partial(process_single_frame, 
                                images=images, 
                                frame_indices=frame_indices, 
                                frame_face_data_epi_sea=frame_face_data_epi_sea,
                                subtitle_data=subtitle_data, 
                                save_render_path=save_render_path, 
                                season=season, 
                                episode=episode,
                                sea_epi=sea_epi)
                
                # Submit tasks for all image indices
                # map will execute the worker function for every index in the range
                executor.map(worker, range(images.shape[0]))
            
    def render_subtitles(self, frame, subtitle_data, frame_id):
        for sub_key, info in subtitle_data['object_list'].items():
            start_frame = info['start_frame']
            end_frame = info['end_frame']
            if start_frame <= int(frame_id) <= end_frame:
                sub_content = info['text']
                frame = self.render_subtitle_impl(frame, sub_content)
                return frame
            elif int(frame_id) > end_frame:
                return frame
        return frame

    def render_subtitle_impl(self, frame, sub_content):
        """
        在 frame 上渲染中文字幕（支持多行）
        
        Args:
            frame (np.ndarray): OpenCV 格式的图像（BGR 或 RGB，但建议传入 RGB）
            sub_content (str): 字幕内容，可以包含换行符 \n
        
        Returns:
            np.ndarray: 添加字幕后的新图像（与输入通道顺序一致）
        """
        # 假设 frame 是 RGB（如果不是，请确保与 PIL 兼容）
        h, w = frame.shape[:2]
        scale = w / 1920.0
        font_size = max(int(50 * scale), 20)  # 防止字体太小
        try:
            font = ImageFont.truetype('./fonts/SimHei.ttf', font_size)
        except IOError:
            # 如果找不到 SimHei，回退到默认字体（但可能不支持中文）
            font = ImageFont.load_default()

        # 转为 PIL Image（注意：PIL 使用 RGB）
        if frame.shape[2] == 3:
            pil_img = Image.fromarray(frame)
        else:
            pil_img = Image.fromarray(frame, mode='L')  # 灰度图处理（一般不会用到）

        draw = ImageDraw.Draw(pil_img)

        # 支持多行字幕
        lines = sub_content.split('\n') if '\n' in sub_content else [sub_content]

        # 计算每行文本尺寸
        line_sizes = []
        max_width = 0
        total_height = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            line_sizes.append((text_w, text_h))
            max_width = max(max_width, text_w)
            total_height += text_h + 10  # 行间距

        total_height -= 10  # 最后一行不需要额外间距

        # 字幕位置：底部居中（留一点边距）
        margin_bottom = int(50 * scale)
        start_y = h - total_height - margin_bottom
        start_y = max(start_y, 0)  # 防止超出图像顶部

        # 绘制每一行（带黑色描边提升可读性）
        stroke_width = max(1, int(2 * scale))
        y_offset = start_y
        for i, line in enumerate(lines):
            text_w, text_h = line_sizes[i]
            x = (w - text_w) // 2

            # 先画黑色描边（多次偏移模拟粗边）
            for dx in (-stroke_width, 0, stroke_width):
                for dy in (-stroke_width, 0, stroke_width):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y_offset + dy), line, fill=(0, 0, 0), font=font)

            # 再画白色正文
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += text_h + 10

        # 转回 numpy array
        return np.array(pil_img)
    
    def time_str_to_seconds(self, time_str):
        """将 'HH:MM:SS.mmm' 格式的时间字符串转为秒数（float）"""
        hms, ms = time_str.split('.')
        h, m, s = map(int, hms.split(':'))
        total_sec = h * 3600 + m * 60 + s + int(ms) / 1000.0
        return total_sec

    def draw_box(self, frame_face_data, img, thred, sea_epi, scale_factor):
        # if scale_factor != 1:
        #     img = self.resize_img(img, scale_factor)
        a = img.shape[1] / 1920

        font_size = max(int(50 * a), 15)  # 防止字体太小
        font = ImageFont.truetype('./fonts/SimHei.ttf', font_size)
        text_bbox = []
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        speaker_path = os.path.join(self.episode_info[sea_epi]['video_path'].split('video_data')[0], 'speaker_data')
        speak_map_path = None
        try:
            with open(os.path.join(speaker_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
                speaker_file = version_data['speaker_recognition_step']['active_version']
                speak_map_path = os.path.join(speaker_path, speaker_file, '.workspace/new_roles_map.json')
        except:
            pass

        if speak_map_path is not None:
            with open(speak_map_path) as f:
                mapping_name = json.load(f)
        else:
            mapping_name = {}

        episode = sea_epi.split('_')[1].zfill(4)
        for face in frame_face_data:
            role_name = face["id_list"]['0']['name']
            confidence = float(face["id_list"]['0']['score'])
            bbox = list(face["geometry"]['active_frame_list']['0']['box'].values())
            if confidence < thred or f'{episode}_pro' in role_name:
                continue
            if '少年' in role_name:
                role_name = '少年' + role_name.replace('少年','')
            elif '幼年' in role_name:
                role_name = '幼年' + role_name.replace('幼年', '')
            elif '童年' in role_name:
                role_name = '童年' + role_name.replace('童年', '')
            elif '成年' in role_name:
                role_name = '成年' + role_name.replace('成年', '')  
            elif '青年' in role_name:
                role_name = '青年' + role_name.replace('青年', '')

            role_name = role_name.replace('（）','').replace('()','')
            if '新角色' in role_name:
                if role_name in mapping_name.keys():
                    role_name = mapping_name[role_name]
                else:
                    role_name = '其他'

            if role_name in replace_names.keys():
                # print(f'*****{role_name}***{replace_names[role_name]}******')
                role_name = replace_names[role_name]
            if 'New Role' in role_name or '新角色' in role_name:
                if role_name in mapping_name.keys():
                    role_name = mapping_name[role_name]
                else:
                    role_name = 'Others'

            left, top, right, bottom = font.getbbox(role_name)
            text_width = right - left
            text_height = bottom - top  

            bbox = [max(0, _) * scale_factor  for _ in bbox]
            bbox[1] = max(text_height+5, bbox[1])
            bbox[2] = min(img.shape[1],  bbox[2])
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                continue

            draw.rectangle(bbox, outline=(255, 0, 0), width=math.ceil(4*scale_factor))  # 这里使用红色线条，线条宽度为2

            text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) // 2
            text_y = bbox[1] - text_height - 5

            Bx1, By1, Bx2, By2 = text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5
            # 处理名字框重叠
            for box_old in text_bbox:
                attemp = 0
                iou = self.calculate_iou(box_old, [Bx1, By1, Bx2, By2])
                while iou > 0.05 and attemp < 5:
                    By1 -= 20
                    By2 -= 20
                    text_y -= 20
                    attemp += 1
                    iou = self.calculate_iou(box_old, [Bx1, By1, Bx2, By2])

            # 绘制文字背景矩形
            draw.rectangle([(Bx1, By1), (Bx2, By2)], fill=(255, 0, 0))
            text_bbox.append([Bx1, By1, Bx2, By2])
            # 绘制文字
            draw.text((text_x, text_y), role_name, fill=(255, 255, 255), font=font)
        return np.array(pil_img)
    
    def resize_img_pil(self, img):
        # 将 numpy array 转为 PIL Image
        pil_img = Image.fromarray(img)

        # 计算新尺寸
        new_size = (int(pil_img.width * self.scale_factor), int(pil_img.height * self.scale_factor))

        # 缩放（使用高质量重采样）
        resized_pil = pil_img.resize(new_size, Image.Resampling.LANCZOS)  # 或 Image.BILINEAR

        # 转回 numpy array
        resized_img = np.array(resized_pil)
        return resized_img

    def resize_img(self, img):
        w, h = img.size
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        return img.resize(new_size, Image.Resampling.LANCZOS)
    
    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的 IoU（交并比）
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        
        Returns:
            iou: float, 交并比，范围 [0, 1]
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集区域的坐标
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # 计算交集面积
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # 计算各自面积
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 避免除零
        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou
    
    def get_speaker(self, time, sample_res, has_season=False):
        speaker_save_info = {}
        for sample_item in sample_res:
            video_key = sample_item[0]
            episode = video_key.split('_')[1].zfill(4)
            season = video_key.split('_')[0]
            images = sample_item[1]
            frame_indices = sample_item[2]
            time_ = time.replace(':', '-')
            time_ = time_.replace('#', '-')
            speaker_path = os.path.join(self.episode_info[video_key]['video_path'].split('video_data')[0], 'speaker_data')
            with open(os.path.join(speaker_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
                speaker_file = version_data['speaker_recognition_step']['active_version']
                speaker_file_path = os.path.join(speaker_path, speaker_file, episode+'.json')
                with open(speaker_file_path, 'r', encoding='utf-8') as file:
                    speaker_data = json.load(file)
                    for im_id in range(images.shape[0]):
                        frame_id = frame_indices[im_id]
                        if has_season == False:
                            frame_key = episode + '_' + str(frame_id) 
                        else:
                            frame_key = season + '_' + episode + '_' + str(frame_id)
                        speaker_save_info[frame_key] = {}
                        for sub_key, info in speaker_data['object_list'].items():
                            start_frame = info['start_frame']
                            end_frame = info['end_frame']
                            if start_frame <= int(frame_id) <= end_frame:
                                speaker_save_info[frame_key] = {
                                    'role': info['role'],
                                    'text': info['text']
                                }

        sampled_speaker_path = os.path.join(self.image_path, time_, 'spkeaker.json')
        with open(sampled_speaker_path, "w", encoding="utf-8") as f:
            json.dump(speaker_save_info, f, ensure_ascii=False, indent=4)

    def evaluation(self, result):
        gts = 0
        tps = 0
        total_tokens = 0
        for q_i, q_info in result.items():
            an = q_info['Q&A']['Answer']
            if 'result' not in q_info['Q&A'].keys():
                continue
            gts += 1
            pred = q_info['Q&A']['result']
            if pred == an:
                tps += 1
            total_tokens += q_info['Q&A']['total_tokens']

        acc = (tps / gts) * 100
        tokens_per_q = (total_tokens / gts)
        print('model: {}, question num: {}, accuracy: {:.2f}, avg tokens per question {:.2f}'.format(
                                                                    self.short_model_name, 
                                                                    gts, acc, tokens_per_q))

def main(args):
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',
                         type=str, 
                         default='./results/img_data', 
                         help='Another example parameter')
    parser.add_argument('--drama',
                         type=str, 
                         default='ren_shi_jian', 
                         help='Another example parameter')
    parser.add_argument('--model_name', 
                        type=str, 
                        default="gpt-4.1",
                        help='[gpt-4.1, gpt-5, gpt-4o, gemini-2.5-pro, qwen2.5-vl-72b, qwen3-vl-32b,' \
                        'qwen3-vl-235b, glm4.6v, intern-vl-38b, intern-vl-241b, qwen2.5-vl-7b]')
    parser.add_argument('--question_path', 
                        type=str, 
                        default="./questions/ren_shi_jian/detail_selected_4_files_rules_filted_match_human_check_local_newformat.json",
                        help='question path')
    parser.add_argument('--sample_frames',
                         type=int,
                         default=64, 
                         help='Another example parameter')
    parser.add_argument('--video_path',
                         type=str, 
                         default='D:/data/ren_shi_jian', 
                         help='Another example parameter')
    parser.add_argument("--scale_factor",
                        type=float,
                        default=1,
                        help="缩放图片尺寸")
    parser.add_argument('--render_subtitle',
                         type=bool, 
                         default=False, 
                         help='Another example parameter')
    parser.add_argument('--num_workers',
                         type=int,
                         default=16, 
                         help='Another example parameter')
    parser.add_argument('--use_render_info', 
                        action='store_true', 
                        default=False, 
                        help='Another example parameter')
    parser.add_argument('--parallel',
                         action='store_true', 
                         default=False, 
                         help='Another example parameter')
    parser.add_argument('--media_resolution',
                         type=str, 
                         default='low', 
                         help='[low,medium,unspecified,high,ultra_high]')
    parser.add_argument('--level',
                         type=str, 
                         default='L2', 
                         help='[L2, L3, L4, L5]')
    parser.add_argument('--caption_path',
                         type=str, 
                         default='./captions/ren_shi_jian/ren_shi_jian_hierarchical_caption_Hiera_1023_3API_briefadaptlength_20251028.json', 
                         help='Another example parameter')
    parser.add_argument('--only_gen_img',
                         action='store_true', 
                         default=False, 
                         help='only sample imgage')
    parser.add_argument('--group_4x4',
                         action='store_true',
                         default=False,
                         help='group images in 4x4 grid')
    parser.add_argument('--group_2x2',
                         action='store_true',
                         default=False,
                         help='group images in 2x2 grid')
                

    args = parser.parse_args()
    engine = LLMInference(args)
    engine.read_caption_info()
    engine.read_question_info()

    if args.only_gen_img == False:
        time1 = time.time()
        if args.parallel:
            result = engine.parallel_inference()
        else:
            result = engine.single_inference()
        time2 = time.time()
        print(f'cost: {time2 - time1} s')
        engine.evaluation(result)

if __name__ == "__main__":
    main(sys.argv[1:])