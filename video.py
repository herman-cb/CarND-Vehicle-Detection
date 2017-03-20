from utils import *
from moviepy.editor import VideoFileClip
from IPython.display import HTML

test_output = 'test.mp4'
# clip = VideoFileClip('test_video.mp4')
clip = VideoFileClip('project_video.mp4')
test_clip=clip.fl_image(process_image)
test_clip.write_videofile(test_output, audio=False)

