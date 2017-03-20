from utils import *
from training_data import TrainingData
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

trainingData = TrainingData()

paths = trainingData.get_file_paths()

cars = paths["cars"]
non_cars = paths["non-cars"]
car_ind = np.random.randint(0, len(cars))
non_car_ind = np.random.randint(0, len(non_cars))

car_image = mpimg.imread(cars[car_ind])
non_car_image = mpimg.imread(non_cars[non_car_ind])

color_space = 'RGB'
orient = 6
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True

car_features, car_hog_image = single_img_features(car_image, color_space=color_space,
                                                 spatial_size = spatial_size,
                                                 hist_bins = 16, orient=orient,
                                                 pix_per_cell = pix_per_cell,
                                                 cell_per_block=cell_per_block,
                                                 hog_channel=hog_channel,
                                                  hog_feat=hog_feat,
                                                  spatial_feat=spatial_feat,
                                                  hist_feat=hist_feat,
                                                  vis=True
                                                 )
non_car_features, non_car_hog_image = single_img_features(non_car_image, color_space=color_space,
                                                 spatial_size = spatial_size,
                                                 hist_bins = 16, orient=orient,
                                                 pix_per_cell = pix_per_cell,
                                                 cell_per_block=cell_per_block,
                                                 hog_channel=hog_channel,
                                                  hog_feat=hog_feat,
                                                  spatial_feat=spatial_feat,
                                                  hist_feat=hist_feat,
                                                  vis=True
                                                 )
images = [car_image, car_hog_image, non_car_image, non_car_hog_image]
titles = ['car image', 'car hog image', 'non car image', 'non car hog image']
fig = plt.figure(figsize=(12,3))
visualize(fig, 1, 4, images, titles)
print("Showing the plot now")
plt.interactive(False)
plt.show()