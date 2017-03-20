# Smart computation of hog features

import glob
from train import *
import pickle

example_images = glob.glob("vehicle_det_examples/*")

svc = pickle.load(open("svc.pkl", "rb"))
X_scaler = pickle.load(open("X_scaler.pkl", "rb"))

out_images = []
out_maps = []
out_titles = []
out_boxes = []

ystart = 400
ystop = 656
scale = 1.5
for img_src in example_images:
    img_boxes = []
    t = time.time()
    count = 0
    img = mpimg.imread(img_src)
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
            count += 1
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

    print("Time taken = {} seconds for {} windows".format(time.time() - t, count))

    out_images.append(draw_img)

    out_titles.append(img_src[21:])
    out_titles.append(img_src[21:])

    out_images.append(heatmap)
    out_maps.append(heatmap)
    out_boxes.append(img_boxes)

fig = plt.figure(figsize=(12, 24))
visualize(fig, 8, 2, out_images, out_titles)
plt.show()



