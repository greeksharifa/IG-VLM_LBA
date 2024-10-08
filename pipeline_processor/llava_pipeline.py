"""
 Copyright (c) 2024, Deep Representation Learning Research Group, Seoul National University.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import uuid
import math
import pandas as pd

from tqdm import tqdm

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_processor.llava2_model_processor import *
from vision_processor.fps_gridview_processor import *
from .record import *
from data.STAR.video_clip import clip_video


class LlavaPipeline:
    def __init__(
        self,
        model_name,
        path_qa,
        path_video_file_format,
        dir="./llava_pipeline_result/",
    ):
        self.model_name = "liuhaotian/" + model_name
        self.path_qa = path_qa
        self.path_dir = dir
        self.path_result = dir
        self.path_video_file_format = path_video_file_format
        self.error_video_name = []
        self.make_video_file_list()
        self.load_model()

    def make_video_file_list(self):
        self._load_qa_file()
        self.df_qa["path_video"] = self.df_qa.apply(
            lambda x: (self.path_video_file_format % (x["video_name"])), axis=1
        )

    def load_model(self):
        self.model = Llava2Processor(self.model_name)
        self.model.load_model()

    def set_component(
        self,
        user_prompt,
        frame_fixed_number=6,
        func_user_prompt=lambda prompt, row: prompt % (row["question"]),
        calculate_max_row=lambda x: round(math.sqrt(x)),
    ):

        if not hasattr(self, "model"):
            raise AttributeError("Model is not loaded. Please call load_model()")

        self.frame_fixed_number = frame_fixed_number
        self.user_prompt = user_prompt
        self.func_user_prompt = func_user_prompt
        self.calculate_max_row = calculate_max_row

        self.fps_data_processor = FpsDataProcessor(
            save_option=SaveOption.IMAGE,
            calcualte_max_row=self.calculate_max_row,
            frame_fixed_number=self.frame_fixed_number,
        )

        extra_dir = "ffn=%s/" % (str(self.frame_fixed_number))
        self._make_directory(extra_dir)

    def do_pipeline(self, num_data, sub_qa_index, num_sub_qa_select):
        print("start pipeline")
        # import pdb; pdb.set_trace()

        for idx, row in tqdm(self.df_qa.iterrows()):
            if num_data != -1 and idx >= num_data: 
                break
            # if idx < 7839:
            #     continue
            
            question_id = str(row["question_id"])
            video_path = row["path_video"]
            ts = row["ts"] if "ts" in row else None
            video_extensions = ["avi", "mp4", "mkv", "webm", "gif"]
            
            if not os.path.exists(video_path) and 'TVQA' not in video_path:
                base_video_path, _ = os.path.splitext(video_path)
                for ext in video_extensions:
                    temp_path = f"{base_video_path}.{ext}"
                    if os.path.exists(temp_path):
                        video_path = temp_path
                        break

            """
            if not os.path.exists(self._make_file_path(question_id)) or \
                (sub_qa_index != "base" and ('%' in str(row["sub_question_" + sub_qa_index]) or '%' in str(row["sub_answer_" + sub_qa_index]))) or \
                video_path in [
                    './data/TVQA/videos/house_s08e19_seg02_clip_20.mp4',
                    './data/TVQA/videos/castle_s07e23_seg02_clip_14.mp4',
                    './data/TVQA/videos/house_s08e16_seg02_clip_08.mp4',
                    './data/TVQA/videos/s01e16_seg02_clip_01.mp4',
                    './data/TVQA/videos/castle_s04e16_seg02_clip_21.mp4',
                ]:
            """
            if not os.path.exists(self._make_file_path(question_id)):
                try:
                    # import pdb; pdb.set_trace()
                    image_data = self.fps_data_processor.process([video_path], ts)
                    
                    user_prompt = self.func_user_prompt(self.user_prompt, row)
                    
                    if sub_qa_index != "base":
                        context = 'Context:\n'
                        
                        for j in range(num_sub_qa_select):
                            sq_idx = (int(sub_qa_index) + j) % 5
                            sq_idx = str(sq_idx)
                            context += f'{row["sub_question_" + sq_idx].rstrip("?")}? {row["sub_answer_" + sq_idx]}.\n'
                            
                        user_prompt = context + user_prompt
                            
                        # user_prompt = f'Context: {row["sub_question_" + sub_qa_index].rstrip("?")}? {row["sub_answer_" + sub_qa_index]}.\n' + user_prompt

                    if idx == 0:
                        print('*' * 40 + f'\nquestion_id: {question_id} \nuser_prompt:\n{user_prompt}\n' + '*' * 40)

                    answer, confidence_score = self.model.infer_and_save(
                        user_prompt=user_prompt,
                        raw_image=image_data,
                    )
                    if -1 != answer:
                        # answer = answer[0]
                        self.write_result_file(question_id, answer)
                        self.write_result_file(f'{question_id}_confidence_score', str(confidence_score))
                    else:
                        self.error_video_name.append(video_path)
                    # print(f'question_id: {question_id} \t answer: {answer[0]}')
                except Exception as e:
                    print(f'########Exception########:\nidx: {idx} \t question_id: {question_id} \t video_path: {video_path}')
                    import traceback
                    traceback.print_exc()
                    import pdb; pdb.set_trace()
                    continue

        return self.merge_qa_and_answer()

    def write_result_file(self, question_id, answer, extension=".txt"):
        file_path = self._make_file_path(question_id, extension)
        with open(file_path, "w") as file:
            file.write(answer)

    def _make_file_path(self, question_id, extension=".txt"):
        return os.path.join(self.path_result, question_id + extension)

    def _load_qa_file(self):
        try:
            self.df_qa = pd.read_csv(self.path_qa, index_col=0)
        except Exception as e:
            print(e)
            raise Exception("not valid qa files")

    def _make_directory(self, extra_dir):
        self.path_result = os.path.join(self.path_dir, extra_dir)
        os.makedirs(self.path_result, exist_ok=True)

    def merge_qa_and_answer(self):
        print("start merge_qa_and_answer")

        self.df_qa["pred"] = None
        path_merged = os.path.join(self.path_result, "result.csv")

        if not os.path.exists(path_merged):
            for idx, row in self.df_qa.iterrows():
                question_id = str(row["question_id"])
                file_path = self._make_file_path(
                    question_id,
                )
                confidence_score_path = self._make_file_path(
                    f'{question_id}_confidence_score',
                )
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as file:
                            file_contents = file.read()
                        self.df_qa.loc[idx, "pred"] = file_contents
                        with open(confidence_score_path, "r") as file:
                            file_contents = file.read()
                        self.df_qa.loc[idx, "confidence_score"] = file_contents
                    except Exception as e:
                        print(file_path)
                        raise (e)

            self.df_qa.to_csv(path_merged)
        else:
            self.df_qa = pd.read_csv(path_merged, index_col=0)

        return self.df_qa, path_merged
