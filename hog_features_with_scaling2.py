import glob
from utils import *
from train import *
out_images = []
out_maps = []
out_titles = []
out_boxes = []

example_images = glob.glob("vehicle_det_examples/*")
X_scaler = pickle.load(open("X_scaler.pkl", "rb"))
svc = pickle.load(open("svc.pkl", "rb"))

ystart = 400
ystop = 656
scale = 1.5
for img_src in example_images:
    print("Going through {}".format(img_src))
    img = mpimg.imread(img_src)
    out_img, heatmap = find_cars(img, scale, X_scaler, svc)
    print("Unique heatmap values = {}".format(np.unique(heatmap)))
    heatmap = apply_threshold(heatmap, 6)
    print("max heat = {}, min heat = {}".format(np.max(heatmap), np.min(heatmap)))
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    out_images.append(draw_img)
    out_images.append(heatmap)
    out_titles.append(img_src[21:])
    out_titles.append(img_src[21:])

print("Number of out_images = {}".format(len(out_images)))
print("Number of out_titles = {}".format(len(out_titles)))
fig = plt.figure(figsize=(12, 24))
visualize(fig, 8, 2, out_images, out_titles)
plt.show()