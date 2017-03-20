import os
import glob

class TrainingData():
    def __init__(self, vehicle_dir="vehicles/", non_vehicle_dir="non-vehicles/"):
        self.vehicle_dir = vehicle_dir
        self.non_vehicle_dir = non_vehicle_dir
        self.cars = []
        self.non_cars = []

    def get_file_paths(self):

        image_types = os.listdir(self.vehicle_dir)

        for imtype in image_types:
            self.cars.extend(glob.glob(self.vehicle_dir + imtype + "/*"))

        with open("cars.txt", "w") as f:
            for path in self.cars:
                f.write(path + "\n")

        image_types = os.listdir(self.non_vehicle_dir)

        for imtype in image_types:
            self.non_cars.extend(glob.glob(self.non_vehicle_dir + imtype + "/*"))

        with open("non-cars.txt", "w") as f:
            for path in self.non_cars:
                f.write(path + "\n")

        return {"cars": self.cars, "non-cars":self.non_cars}

if __name__ == "__main__":
    training_data = TrainingData()
    paths = training_data.get_file_paths()
    print("Number of car images found = {}".format(len(paths["cars"])))
    print("Number of non car images found = {}".format(len(paths["non-cars"])))


