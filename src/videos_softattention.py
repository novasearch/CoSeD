import json
from diffusers import DiffusionPipeline
from diffusers.utils import load_image, export_to_video
import torch
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import sys

output_path = "videos/"

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1")

pipeline.enable_model_cpu_offload()

file = json.load(open("videos.json"))


for key, value in file.items():
    print(key, value)
    for i, file_path in enumerate(value):
        method = file_path.split("/")[0]
        image = Image.open(file_path)
        image = image.resize((1024, 576))
        generator = torch.manual_seed(42)
        frames = pipeline(image, decode_chunk_size=8, generator=generator).frames[0]
        export_to_video(frames, output_path + key + "_" + str(i) + ".mp4", fps=7)
    clips = [VideoFileClip(output_path + key + "_" + str(i) + ".mp4") for i in range(len(value))]
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path + key + ".mp4", codec="libx264", fps=7)
