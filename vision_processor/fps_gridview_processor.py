import sys
import os
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pipeline_processor.record import *
from .fps_extractor import *
from .gridview_generator import *
from .video_validation import *


class FpsDataProcessor:
    def __init__(
        self,
        calcualte_max_row=lambda x: round(math.sqrt(x)),
        save_option=SaveOption.IMAGE,
        frame_fixed_number=6,
    ):
        self.calculate_max_row = calcualte_max_row
        self.frame_fixed_number = frame_fixed_number
        self.save_option = save_option

    def process(self, video_path, ts=None):
        if 'TVQA' in video_path[0]:
            rlt_fps_extractor = load_tvqa_frames(video_path[0], ts)
        else:
            fps_extractor = FpsExtractor(video_path)
            
        grid_view_creator = GridViewCreator(
            self.calculate_max_row,
        )

        try:
            # <class 'numpy.ndarray'> shape=(6, 480, 320, 3)
            if 'TVQA' in video_path[0]:
                pass   # rlt_fps_extractor = rlt_fps_extractor
            else:
                rlt_fps_extractor = fps_extractor.save_data_based_on_option(
                    SaveOption.NUMPY,
                    frame_fixed_number=self.frame_fixed_number,
                    ts=ts,
                )
            # <class 'PIL.Image.Image'> size=(640, 1440)
            rlt_grid_view_creator = grid_view_creator.post_process_based_on_options(
                self.save_option, rlt_fps_extractor
            )
        except Exception as e:
            print("Exception : %s on %s" % (str(e), str(video_path)))
            return -1

        return rlt_grid_view_creator


# ex. video_path='/data/TVQA/videos/friends_s03e17_seg02_clip_01.mp4'
# ex. ts='20.16-25.12'
def load_tvqa_frames(video_path, ts=None):
    vid = video_path.split("/")[-1].split(".")[0] # ex. friends_s03e17_seg02_clip_01
    vis_root = '/data/video_datasets/TVQA/frames_hq/'
    
    dir_path = os.path.join(vis_root, 
                            f"{vid.split('_')[0]}_frames" if vid.count('_') == 4 else f"bbt_frames",
                            vid)
    image_paths = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))
    
    start_time, end_time = ts.split("-")
    try:
        start_frame = int(float(start_time) * 3)
        end_frame = int(float(end_time) * 3)
        if 0 <= start_frame < end_frame <= len(image_paths):
            image_paths = image_paths[start_frame:end_frame]
        elif start_frame >= len(image_paths):
            image_paths = [image_paths[-1]]
        elif end_frame <= 0:
            image_paths = [image_paths[0]]
        else:
            image_paths = [image_paths[start_frame]]
    except: # ts is not valid. ex) ts='NaN-NaN'
        pass

    idxs = np.linspace(0, len(image_paths)-1, 6, dtype=int) # ffn=6
    
    image_paths = [image_paths[idx] for idx in idxs]
        
    # load 6 frames from image_dir
    frames = []
    for image_path in image_paths:
        frame = Image.open(image_path)
        frames.append(np.array(frame))
    
    return np.stack(frames, axis=0)
    


def main():

    video_name = "rlQ2kW-FvMk_66_79.mp4"

    fps_data_processor = FpsDataProcessor(
        save_option=SaveOption.IMAGE,
        frame_fixed_number=6,
    )
    print(vars(fps_data_processor))
    rlt = fps_data_processor.process(["example", video_name])
    print(rlt)

    rlt.save("./example/imagegrid_sample/%s.jpg" % (video_name.split(".")[0]))


if __name__ == "__main__":
    main()
