import os
import csv
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage import io

def read_images(path):
    return io.imread(path).astype(np.float32).flatten()

def main():
    files = [os.path.join("./images", file) for file in os.listdir("./images")]
    
    pool = mp.Pool(8)
    X = np.array(pool.map(read_images, files))
    pool.close()
    pool.join()

    X /= 255.0
    X = (X - 0.5) / 0.5

    mean = np.mean(X, axis=0)

    pca = PCA(n_components=10, random_state=0)
    X_reduced = pca.fit_transform(X-mean)

    kmeans = KMeans(n_clusters=2, max_iter=3000, random_state=0)
    kmeans.fit(X_reduced)

    df = pd.read_csv("./test_case.csv")
    IDs, name1, name2 = np.array(df['id']), np.array(df['image1_name']), np.array(df['image2_name'])

    with open("./output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])

        for id, n1, n2 in zip(IDs, name1, name2):
            if kmeans.labels_[n1-1] == kmeans.labels_[n2-1]:
                writer.writerow([id, 1])
            else:
                writer.writerow([id, 0])
    
if __name__ == "__main__":
    main()