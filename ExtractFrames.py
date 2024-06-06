"""
-The given code extracts all the frames for the entire dataset and saves these frames in the folder of the video clips.
-Kindly have ffmpeg (https://www.ffmpeg.org/) (all credits) in order to successfully execute this script.
-The script must in the a same directory as the Dataset Folder.
"""

import os
import subprocess

# The folder containing the DAiSEE dataset
data_dir = "Data"
# The folder in which the extracted frames will be saved
output_frames_dir = "ExtractedImages"
if not os.path.exists(output_frames_dir):
  os.makedirs(output_frames_dir)


def extract_frames(video_filepath, image_name_prefix):
    input_path = video_filepath
    # The first frame of the video
    output_path_first_frame = f"{output_frames_dir}/{image_name_prefix}_first.jpg"
    subprocess.check_output(
        f'ffmpeg -i "{input_path}" -ss 0 -frames:v 1 "{output_path_first_frame}" -hide_banner',
        shell=True,
    )
    output_path_last_frame = f"{output_frames_dir}/{image_name_prefix}_last.jpg"
    # One of the last frames of the video
    subprocess.check_output(
        f'ffmpeg -sseof -3 -i "{input_path}" -frames:v 1 "{output_path_last_frame}" -hide_banner',
        shell=True,
    )


dataset = os.listdir(data_dir)
for ttv in dataset:
    ttv_dir = os.path.join(data_dir, ttv)
    if not os.path.isdir(ttv_dir):
        continue
    users = os.listdir(ttv_dir)
    for user in users:
        user_dir = os.path.join(ttv_dir, user)
        if not os.path.isdir(user_dir):
            continue
        currUser = os.listdir(user_dir)
        for extract in currUser:
            extract_dir = os.path.join(user_dir, extract)
            if not os.path.isdir(extract_dir):
                continue
            first_extract_dir = extract_dir
            break
        # As we want at most 2 frames of an individual in our dataset
        # We can extract multiple frames from a single video, and so if there are multiple clips, we just consider the first one
        first_clip = os.listdir(first_extract_dir)[0]
        print(first_clip[:-4])
        extract_frames(
            os.path.join(first_extract_dir, first_clip),
            first_clip[:-4],
        )

print(
    "================================================================================\n"
)
print("Frame Extraction Successful")