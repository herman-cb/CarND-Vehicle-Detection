import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import pickle

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=True,
                                  feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=False,
                       feature_vector=feature_vec)
        return features


# Downsamples the image
def bin_spatial(img, size=(32, 32)):
    color0 = cv2.resize(img[:, :, 0], size).ravel()
    color1 = cv2.resize(img[:, :, 1], size).ravel()
    color2 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color0, color1, color2))


def color_hist(img, nbins=32):
    # Compute the histogram of the RGB channels separately
    c0_hist = np.histogram(img[:, :, 0], bins=nbins)
    c1_hist = np.histogram(img[:, :, 1], bins=nbins)
    c2_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((c0_hist[0], c1_hist[0], c2_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis=False):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            if vis:
                print
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                           pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    if vis:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    for fname in imgs:
        file_features = []
        image = mpimg.imread(fname)

        file_features = single_img_features(image,
                                            color_space=color_space,
                                            spatial_size=spatial_size,
                                            hist_bins=hist_bins,
                                            orient=orient,
                                            pix_per_cell=pix_per_cell,
                                            cell_per_block=cell_per_block,
                                            hog_channel=hog_channel,
                                            spatial_feat=spatial_feat,
                                            hist_feat=hist_feat,
                                            hog_feat=hog_feat,
                                            vis=False)
        #         concatenated_file_features = np.concatenate(file_features)
        features.append(file_features)
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    window_list = []
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]

    x = x_start_stop[0]
    y = y_start_stop[0]

    while x + xy_window[0] <= x_start_stop[1]:
        while y + xy_window[1] <= y_start_stop[1]:
            window_list.append(((x, y), (x + xy_window[0], y + xy_window[1])))
            y += np.int((1 - xy_overlap[1]) * xy_window[1])
        y = y_start_stop[0]
        x += np.int((1 - xy_overlap[0]) * xy_window[0])

    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.title(i + 1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])


def convert_color(img, conv='RGB2YCrCb'):
    c = {'RGB2YCrCb': cv2.COLOR_RGB2YCrCb,
        'BGR2YCrCb': cv2.COLOR_BGR2YCrCb,
        'RGB2LUV': cv2.COLOR_RGB2LUV}
    return cv2.cvtColor(img, c[conv])


def find_cars(img, scale, X_scaler, svc, heatmaplist=[]):
    ystart = 400
    ystop = 656
    img_boxes = []
    # scale = 1.5

    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:, :, 0])
    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0])))

    ch0 = ctrans_tosearch[:, :, 0]
    ch1 = ctrans_tosearch[:, :, 1]
    ch2 = ctrans_tosearch[:, :, 2]

    nxblocks = (ch0.shape[1] // pix_per_cell) - 1
    nyblocks = (ch0.shape[0] // pix_per_cell) - 1

    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1

    cells_per_step = 2
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog0 = get_hog_features(ch0, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            hog_feat_0 = hog0[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat_1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat_2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat_0, hog_feat_1, hog_feat_2))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (window, window))

            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            test_features_raw = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(test_features_raw)
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ybox_top = np.int(ytop * scale)

                win_draw = np.int(window * scale)

                cv2.rectangle(draw_img, (xbox_left, ybox_top + ystart),
                              (xbox_left + win_draw, ybox_top + win_draw + ystart),
                              (0, 0, 255))
                img_boxes.append(((xbox_left, ybox_top + ystart), (xbox_left + win_draw, ybox_top + win_draw + ystart)))
                heatmap[ybox_top + ystart:ybox_top + win_draw + ystart, xbox_left:xbox_left + win_draw] += 1
    # print("shape before = {}".format(heatmap.shape))
    heatmap[heatmap > 0] = 1
    # print("shape after = {}".format(heatmap.shape))
    # print("Max heatmap = {}, min heatmap = {}".format(np.max(heatmap), np.min(heatmap)))
    heatmaplist.append(heatmap)

    if len(heatmaplist) >= 5:
        heatmap = np.sum(np.stack(heatmaplist)[-5:, :, :], axis=0)
    return draw_img, heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img

def process_image(img, scale=1.5):
    X_scaler = pickle.load(open("X_scaler.pkl", "rb"))
    svc = pickle.load((open("svc.pkl", "rb")))
    out_img, heatmap = find_cars(img, scale, X_scaler, svc)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img

