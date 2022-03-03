import numpy as np
from scipy.io import arff


class DataLoader:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def load_data_file(self, data_path, label_path):
        X = np.loadtxt(self.data_folder_path + data_path, dtype=float)
        Y = np.loadtxt(self.data_folder_path + label_path, dtype=float)
        return X, Y

    def load_arff_file(self):
        data_file = "./data/elecNormNew.arff"
        data, meta = arff.loadarff(data_file)

        data_list_x = []
        data_list_y = []

        for d in data:
            data_point = []
            for i in range(8):
                data_point.append((float)(d[i]))
            data_list_x.append(data_point)
            if d[8] == b'UP':
                data_list_y.append(1)
            else:
                data_list_y.append(0)

        X = np.array(data_list_x)
        Y = np.array(data_list_y)

        X = X.astype(np.float16)

    def load_chess(self):
        return self.load_data_file("./data/driftDatasets/artificial/chess/transientChessboard.data",
                              "./data/driftDatasets/artificial/chess/transientChessboard.labels")

    def load_hyperplane(self):
        return self.load_data_file("./data/driftDatasets/artificial/hyperplane/rotatingHyperplane.data",
                              "./data/driftDatasets/artificial/hyperplane/rotatingHyperplane.labels")

    def load_mixed_drift(self):
        return self.load_data_file("./data/driftDatasets/artificial/mixedDrift/mixedDrift.data",
                              "./data/driftDatasets/artificial/mixedDrift/mixedDrift.labels")

    def load_moving_squares(self):
        return self.load_data_file("./data/driftDatasets/artificial/movingSquares/movingSquares.data",
                              "./data/driftDatasets/artificial/movingSquares/movingSquares.labels")

    def load_interchanging_rbf(self):
        return self.load_data_file("./data/driftDatasets/artificial/rbf/interchangingRBF.data",
                              "./data/driftDatasets/artificial/rbf/interchangingRBF.labels")

    def load_moving_rbf(self):
        return self.load_data_file("./data/driftDatasets/artificial/rbf/movingRBF.data",
                              "./data/driftDatasets/artificial/rbf/movingRBF.labels")

    def load_electricity(self):
        return self.load_data_file("./data/driftDatasets/realWorld/Elec2/elec2_data.dat",
                              "./data/driftDatasets/realWorld/Elec2/elec2_label.dat")

    def load_outdoor(self):
        return self.load_data_file("./data/driftDatasets/realWorld/outdoor/outdoorStream.data",
                              "./data/driftDatasets/realWorld/outdoor/outdoorStream.labels")

    def load_poker(self):
        return self.load_data_file("./data/driftDatasets/realWorld/poker/poker.data",
                              "./data/driftDatasets/realWorld/poker/poker.labels")

    def load_rialto(self):
        return self.load_data_file("./data/driftDatasets/realWorld/rialto/rialto.data",
                              "./data/driftDatasets/realWorld/rialto/rialto.labels")

    def load_toy(self, name):
        return self.load_data_file("./data/driftDatasets/artificial/toy/toy"+name+".data",
                              "./data/driftDatasets/artificial/toy/toy"+name+".label")