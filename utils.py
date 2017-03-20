import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import matplotlib.image as mpimg

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

