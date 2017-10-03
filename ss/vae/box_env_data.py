import numpy as np
import scipy.misc
import glob
import ss.path
import matplotlib.pyplot as plt
import pdb

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def get_files(folder):
    return glob.glob(ss.path.DATADIR + folder + "/*.jpg")

def preprocess(img):
    # return rgb2gray(img)
    return np.logical_or(img[:, :, 1] < 80, img[:, :, 0] > 90)

def get_data(name, N=100000000):
    # name is "boxenv_random" or "boxenv_close"
    print("loading data from", name)
    fs = get_files(name)
    # fs2 = get_files(name)

    data = []
    for f in fs[:N]:
        img = scipy.misc.imread(f)
        img = preprocess(img)
        data.append(img.flatten())

    np.random.shuffle(data)
    data = np.array(data).astype(float)
    # data = (data - np.mean(data)) / np.std(data)
    # data = (data < 0.2).astype(float)
    # for i in range(4):
    #     img = data[i, :].reshape((28 ,28))
    #     plt.imshow(img, cmap = plt.get_cmap('gray'))
    #     plt.show()
    N = len(data) * 9 // 10
    train_data = data[:N, :]
    test_data = data[N:, :]
    test_labels = [0] * len(test_data)
    return train_data, test_data, test_labels

if __name__ == "__main__":
    get_data("boxenv_random", 100)
