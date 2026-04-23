from moviepy import VideoFileClip

def convert_mp4_to_gif(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip = clip.resized(width=480)
    clip.write_gif(output_path, fps=10)

convert_mp4_to_gif("flow.mp4", "flow.gif")