import time
import matplotlib.image as mpimg
import numpy as np
from utils import *
import glob
import pickle
from train import *

svc = pickle.load(open("svc.pkl", "rb"))
X_scaler = pickle.load(open("X_scaler.pkl", "rb"))

search_path = "vehicle_det_examples/*"
example_images = glob.glob(search_path)
images = []
titles = []
y_start_stop = [400, 656]
overlap = 0.5

for img_src in example_images:
    t = time.time()
    img = mpimg.imread(img_src)
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    print("min pixel value = {}, max pixel value = {}".format(np.min(img), np.max(img)))

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=(96, 96), xy_overlap=(overlap, overlap))
    hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient,
                                 pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel,
                                 spatial_feat=spatial_feat,
                                 hist_feat=hist_feat,
                                 hog_feat=hog_feat)
    window_img = draw_boxes(draw_img, hot_windows, color=(0, 0, 255), thick=6)
    images.append(window_img)
    titles.append('')
    print("Time to process one image = {} seconds, over windows = {}".
          format(round(time.time() - t, 2), len(windows)))
fig = plt.figure(figsize=(12, 18), dpi=300)
visualize(fig, 5, 2, images, titles)