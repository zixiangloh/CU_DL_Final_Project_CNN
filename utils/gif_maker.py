import glob
import os
import random
from PIL import Image


def make_gif(frame_folder, out_file):
    file_lists = glob.glob(os.path.join(frame_folder, "*.png"))
    random.shuffle(file_lists)
    file_lists_subset = file_lists[:20]
    frames = [Image.open(image) for image in file_lists_subset]
    # Pick a subset of frames
    frame_one = frames[0]
    frame_one.save(out_file, format="GIF", append_images=frames,
                   save_all=True, duration=1000, loop=0)


if __name__ == "__main__":
    input_frame_path = os.path.join("..", "data", "mnist_clothing", "self_generated", "eight_by_nine_seen_images")
    output_file = os.path.join("..", "docs", "images", "mnist_clothing_samples.gif")
    make_gif(input_frame_path, output_file)
