import os
import sys
import multiprocessing as mp
import numpy as np
from numpy.linalg import svd
from skimage import io

def read_images(path):
    return io.imread(path).astype(np.float32).flatten()

def transform(img):
    img -= np.min(img)
    img /= np.max(img)
    return (img * 255).astype(np.uint8)

def plot_mean_face(faces, save):
    mf = np.mean(faces, axis=0).reshape(600, 600, 3)
    
    if save:
        io.imsave("./mean_face.jpg", transform(mf))
    return mf.flatten()

def plot_topk_eigenface(eigenvectors, k, save):
    for i in range(k):
        ef = eigenvectors[i, :].reshape(600, 600, 3)

        if save:
            io.imsave("./eigenface_{}.jpg".format(i+1), transform(ef))
    return eigenvectors[:k, :]

def topk_percentage(sigma, k):
    total = np.sum(sigma)

    for i in range(k):
        print("Number {}: {}".format(i+1, sigma[i] * 100 / total))
    
def main(argv):
    files = [os.path.join(argv[1], file) for file in os.listdir(argv[1])]
    
    pool = mp.Pool(8)
    X = np.array(pool.map(read_images, files))
    pool.close()
    pool.join()
    
    X_mean = plot_mean_face(X, False)
     
    u, s, vt = svd(X-X_mean, full_matrices=False)
    eigenfaces = plot_topk_eigenface(vt, 5, False)
    #topk_percentage(s, 5)
    

    img = read_images(os.path.join(argv[1], argv[2]))
    img = img - X_mean

    weights = np.dot(img, eigenfaces.T)
    X_recon = X_mean + np.dot(weights, eigenfaces)
    X_recon = X_recon.reshape(600, 600, 3)

    io.imsave(argv[3], transform(X_recon))
    
if __name__ == "__main__":
    main(sys.argv)
