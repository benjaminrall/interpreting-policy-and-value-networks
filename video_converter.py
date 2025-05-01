import os
import imageio
from moviepy import ImageSequenceClip
from PIL import Image
import numpy as np  # add this at the top if not already imported

def extract_frames_and_durations(gif_path):
    frames = []
    durations = []
    with Image.open(gif_path) as img:
        try:
            while True:
                frame = img.copy().convert("RGB")
                frames.append(frame)
                durations.append(img.info.get('duration', 100))  # default to 100ms
                img.seek(img.tell() + 1)
        except EOFError:
            pass
    return frames, durations

def convert_gif_to_mp4_accurate(gif_path, output_path):
    print(f"Processing: {gif_path}")
    frames, durations = extract_frames_and_durations(gif_path)

    # Convert durations to FPS
    avg_duration = sum(durations) / len(durations) / 1000  # Convert ms to seconds
    fps = 1 / avg_duration if avg_duration > 0 else 10  # Fallback FPS

    clip = ImageSequenceClip([np.array(frame) for frame in frames], fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)
    clip.close()

def batch_convert(folder_path):
    for file in os.listdir(folder_path):
        if file.lower().endswith('.gif'):
            gif_path = os.path.join(folder_path, file)
            mp4_path = os.path.join(folder_path, os.path.splitext(file)[0] + ".mp4")
            try:
                convert_gif_to_mp4_accurate(gif_path, mp4_path)
            except Exception as e:
                print(f"Failed to convert {file}: {e}")

if __name__ == "__main__":
    input_folder = "videos/gifs"
    batch_convert(input_folder)