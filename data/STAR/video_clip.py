"""
conda uninstall ffmpeg  
conda install -c conda-forge ffmpeg 

# if Unknown encoder 'libx264': check
ffmpeg -encoders | grep 264

"""

import csv
import subprocess
import os
import json
from tqdm import tqdm
from moviepy.editor import VideoFileClip

#  ffmpeg -y -ss start_time -to end_time -i input_path -codec copy output_path
def clip_video(input_video, output_video, start_time, end_time):
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-i', input_video,
        '-t', str(end_time - start_time),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',
        output_video
    ]
    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        # print(f"Successfully clipped: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"\nError clipping {input_video}: {e}")
        print(f"FFmpeg error output: {e.stderr.decode()}")

def main():
    # csv_file = 'Video_Segments.csv'  # Replace with your CSV file name
    # video_directory = '/data/charades/Charades_v1_480/'   # Replace with the directory containing your videos
    csv_file = '../../data/multiple_choice_qa/STAR.csv'
    video_directory = '/data/video_datasets/STAR/videos_480/'   # Replace with the directory containing your videos
    output_directory = 'videos/'  # Replace with the directory where you want to save the clipped videos
    error_list = []
    processed_set = set()

    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        N = 60206
        for i, row in enumerate(csv_reader):
            # video_id = row['video_id']
            # start_time = float(row['start'])
            # end_time = float(row['end'])
            video_id, start_time, end_time = row['video_name'].split('_')
            start_time, end_time = float(start_time), float(end_time)
            
            # if f"{video_id}_{start_time}_{end_time}.mp4" not in ["P76D1_13.0_20.4.mp4", "P76D1_13.0_20.4.2.mp4", "9KGOL_16.8_24.7.mp4", "ZH4JE_1.4_7.4.mp4"]:
            #     continue
            # else:
            #     print(video_id)

            input_video = os.path.join(video_directory, f"{video_id}.mp4")
            output_video = os.path.join(output_directory, f"{row['video_name']}.mp4")
            if output_video in processed_set:
                print(f"\r{i:5d}/{N}\t\tVideo already processed: {output_video}", end='')
                continue
            processed_set.add(output_video)
            
            if not os.path.exists(input_video):
                print(f"\n{i:5d}/{N}\t\tVideo not found: {input_video}")
            elif os.path.exists(output_video):
                try:
                    VideoFileClip(output_video)
                    print(f"\r{i:5d}/{N}\t\tVideo already exists:", output_video, end='')
                except Exception as e:
                    print(f"\nError opening video file: {output_video}. Reason: {str(e)}")
                    clip_video(input_video, output_video, start_time, end_time)
                    error_list.append(f"Error opening video file: {output_video}")
            else:
                print(f"\n{i:5d}/{N}\t\tClipping {input_video} from {start_time} to {end_time}", end='')
                clip_video(input_video, output_video, start_time, end_time)
                error_list.append(f"Created video file: {output_video}")
            # if i > 5: break
            
    json.dump(error_list, open('errors.json', 'w'), indent=4)
        

if __name__ == "__main__":
    main()
    