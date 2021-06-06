import os
import glob
import math
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import librosa
import random
import colorsys

parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video_path', type=str, default=None,
                    help='Path to the video')

parser.add_argument('--silence_threshold', type=int, default=-40,
                    help='Thereshold below which the audio is considered to be silent')

parser.add_argument('--silence_length', type=int, default=1,
                    help='Split the audio using silences larger than this value (in seconds)')

parser.add_argument('--resolution', type=int, default=8,
                    help='Silence detection resolution e.g 4 --> 0.25 s')

parser.add_argument('--buffer', type=float, default=0.0,
                    help='Adds this to the begining and the end of non-silenct intervals')

parser.add_argument('--flag_threshold', type=int, default=-2,
                    help='Signals above this threshold indicate the discard flag')

parser.add_argument('--remove_silents', action='store_true')

args = parser.parse_args()




def main():
    
    cwd = Path(os.getcwd())
    if not (cwd / 'media').is_dir():
        os.mkdir(cwd / 'media')

    video_path = args.video_path
    print(f"Vidoe path: {video_path}")

    VIDEO_PATH = Path(video_path)
    TEMP_AUDIO_PATH = cwd / 'media' / 'temp_audio.wav'

    clip_name= VIDEO_PATH.stem
    silence_threshold = args.silence_threshold
    silence_length = args.silence_length
    resolution = args.resolution
    buffer = args.buffer
    flag_threshold = args.flag_threshold

    video = VideoFileClip(str(VIDEO_PATH))
    print(f"Video duration: {video.duration} s")
    video.audio.write_audiofile(str(TEMP_AUDIO_PATH))
    fps = video.fps
    sampling_rate = video.audio.fps
    audio_signal, sampling_rate = librosa.load(TEMP_AUDIO_PATH, sr=sampling_rate)
    audio_signal_db = librosa.core.amplitude_to_db(audio_signal, top_db=200)
    audio_duration = audio_signal.shape[0] / sampling_rate
    print(f"Samping rate: {sampling_rate}")
    print(f"Audio duration: {audio_duration} s") 

    # Split the audio_signal into non-silent intervals:
    intervals = librosa.effects.split(audio_signal, top_db=-silence_threshold, frame_length=sampling_rate//silence_length, hop_length=sampling_rate//resolution)
    intervals_seconds = intervals/sampling_rate
    intervals_seconds[:,0] = intervals_seconds[:,0] - buffer
    intervals_seconds[:,1] = intervals_seconds[:,1] + buffer
    intervals_flag = []

    for ind, (sec_in, sec_out) in enumerate(intervals):
        signal_clip = audio_signal_db[sec_in:sec_out]
        if np.max(signal_clip) > flag_threshold:
            intervals_flag.append(ind)

    intervals_to_remove = intervals[intervals_flag]
    intervals_to_keep = []

    intervals_to_keep.append([0, intervals_to_remove[0][0]])
    for i in range(intervals_to_remove.shape[0]-1):
        intervals_to_keep.append([intervals_to_remove[i][1], intervals_to_remove[i+1][0]])
    intervals_to_keep.append([intervals_to_remove[-1][1], audio_signal.shape[0]])
    intervals_to_keep = np.array(intervals_to_keep) / sampling_rate

    _create_edl(VIDEO_PATH, fps, intervals_to_keep)
    


# Helper functions:

def _time_stamp(frames, fps=30):
    """Convert frames (float) to the timestamp format: HH:MM:SS:FF
    """
    hour = int(frames // (3600*fps))
    frames %= (3600*fps)
    minutes = int(frames // (60*fps))
    frames %= (60*fps)
    seconds = int(frames // fps)
    frames %= fps
    frames = int(frames)
    return f"{hour:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def _create_edl(VIDEO_PATH, fps, intervals_to_keep):
    """# Create an EDL based on the flagged intervals
    # EDL format (https://www.niwa.nu/2013/05/how-to-read-an-edl/):

    001  AX  V  C  CLIP_IN_1 CLIP_OUT_1 TIMELINE_BEGIN_1 TIMELINE_END_1
    001  AX  V  C  CLIP_IN_2 CLIP_OUT_2 TIMELINE_BEGIN_2 TIMELINE_END_2

    TIMELINE_BEGIN_1 = 00:00:00:00
    (TIMELINE_END - TIMELINE_BEGIN) = (CLIP_OUT - CLIP_IN)
    TIMELINE_BEGIN_2 = TIMELINE_END_1
    """

    with open(f"{VIDEO_PATH}.{fps:0.2f}.edl", "w+") as f:
        f.write(f"TITLE: {VIDEO_PATH.name}\n")
        f.write("FCM: NON-DROP FRAME\n\n")
        time_line_b = 0
        for ind, (sec_in, sec_out) in enumerate(intervals_to_keep):
            frames_in = math.floor(sec_in*fps)
            frames_out = math.floor(sec_out*fps)
            time_line_e = time_line_b + frames_out - frames_in
            f.write(f"{(ind+1):03d}  AX  V     C  {_time_stamp(frames_in, fps)} {_time_stamp(frames_out, fps)} {_time_stamp(time_line_b, fps)} {_time_stamp(time_line_e, fps)}\n")
            # f.write(f"* FROM CLIP NAME: {VIDEO_PATH.name}\n\n")
            f.write(f"{(ind+1):03d}  AX  AA    C  {_time_stamp(frames_in, fps)} {_time_stamp(frames_out, fps)} {_time_stamp(time_line_b, fps)} {_time_stamp(time_line_e, fps)}\n")
            # f.write(f"* FROM CLIP NAME: {VIDEO_PATH.name}\n\n")
            f.write(f"{(ind+1):03d}  AX  NONE  C  {_time_stamp(frames_in, fps)} {_time_stamp(frames_out, fps)} {_time_stamp(time_line_b, fps)} {_time_stamp(time_line_e, fps)}\n")
            # f.write(f"* FROM CLIP NAME: {VIDEO_PATH.name}\n\n")
            time_line_b = time_line_e

    print(f"EDL was created successfully!")

if __name__ == '__main__':
    main()