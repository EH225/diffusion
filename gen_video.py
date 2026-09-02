# -*- coding: utf-8 -*-
"""
This module can be used to generate timelapse videos of the samples produced during model training.
The required packages needed to run this module are installed to the diffusion-video conda env.

Run this module using:
    conda activate diffusion-video
    cd path/to/project
    python gen_video.py

"""
from pathlib import Path
import re

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

def generate_samples_video(samples_dir: str, fps: int = 10):
    """
    Generates a video time lapse using the samples stored in a given results directory.
    """
    print(f"Working on samples_dir: {samples_dir}")
    output_path = os.path.join(samples_dir, "samples_timelapse.mp4")
    files = [x for x in os.listdir(samples_dir) if x.endswith(".png")]
    files = sorted(files, key=lambda x: int(x.split("_")[2]))

    with imageio.get_writer(output_path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for file in files:
            image = Image.open(os.path.join(samples_dir, file)).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Extract the step number from e.g. step_001000.png
            step = int(file.split("_")[2])
            text = f"Step {step:,}"

            # Draw a translucent-ish black box behind the text
            bbox = draw.textbbox((0, 0), text)
            padding = 12

            x = image.width - (bbox[2] - bbox[0]) - 2 * padding - 20
            y = 20

            draw.rectangle((x, y, image.width - 20, y + (bbox[3] - bbox[1]) + 2 * padding,),
                           fill=(0, 0, 0))
            draw.text((x + padding, y + padding), text, fill="white")
            writer.append_data(np.asarray(image))
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    for dataset in ["cifar10", "celebA", "afhq64", "afhq128"]:
        all_samples_dir = os.path.join(CURRENT_DIR, "results", f"ddpm_{dataset}", "samples")
        for folder in os.listdir(all_samples_dir):
            if folder != "latest":
                generate_samples_video(os.path.join(all_samples_dir, folder))

