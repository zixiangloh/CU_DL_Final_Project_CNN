import glob
import os
import random
from PIL import Image


def make_gif(frame_folder, out_file):
    frames = [Image.open(image) for image in glob.glob(os.path.join(frame_folder, "*.png"))]
    random.shuffle(frames)
    # Pick a subset of frames
    frames_subset = frames[:20]
    frame_one = frames_subset[0]
    frame_one.save(out_file, format="GIF", append_images=frames,
                   save_all=True, duration=1000, loop=0)


if __name__ == "__main__":
    input_frame_path = os.path.join("..", "data", "biased_cars_blurred", "GREEN_CARS", "train", "images")
    output_file = os.path.join("..", "docs", "images", "biased_cars_blurred_samples.gif")
    make_gif(input_frame_path, output_file)
