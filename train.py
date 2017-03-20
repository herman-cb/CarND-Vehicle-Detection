from utils import *
import time
import numpy as np
from training_data import TrainingData
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle

trainingData = TrainingData()

paths = trainingData.get_file_paths()

cars = paths["cars"]
non_cars = paths["non-cars"]

# Train the classifier


def train():
    t = time.time()
    n_samples = 1000
    random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars #np.array(cars)[random_idxs]
    random_idxs = np.random.randint(0, len(non_cars), n_samples)
    test_non_cars = non_cars #np.array(non_cars)[random_idxs]


    car_features = extract_features(test_cars, color_space=color_space,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,
                                   spatial_feat=spatial_feat,
                                   hist_feat=hist_feat,
                                   hog_feat=hog_feat)

    non_car_features = extract_features(test_non_cars, color_space=color_space,
                                   spatial_size=spatial_size,
                                   hist_bins=hist_bins,
                                   orient=orient,
                                   pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,
                                   spatial_feat=spatial_feat,
                                   hist_feat=hist_feat,
                                   hog_feat=hog_feat)

    print("Time to compute features = {} seconds".format(round(time.time()-t, 2)))

    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
                                                        test_size=0.1, random_state=rand_state)
    print("Using {} orientations, {} pixels per cell, {} cells per block, {} histogram bins, and {} spatial sampling".format(
    orient, pix_per_cell, cell_per_block, hist_bins, spatial_size))
    print("Feature vector length = {}".format(len(X_train[0])))

    svc = LinearSVC()

    t = time.time()
    svc.fit(X_train, y_train)

    print("Time to train SVC = {} seconds".format(round(time.time()-t, 2)))
    print("Test accuracy = {}".format(round(svc.score(X_test, y_test), 4)))

    pickle.dump(svc, open("svc.pkl", "wb"))
    pickle.dump(X_scaler, open("X_scaler.pkl", "wb"))

if __name__ == "__main__":
    train()