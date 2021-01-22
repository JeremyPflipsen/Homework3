from skimage import io
from sklearn import cluster
from scipy import stats
import numpy as np
from numpy import multiply as mul
import matplotlib.pyplot as plt


def EMG(file, k, flag):
    # import file and convert to 2D array
    img = io.imread(file)
    img = img / 255

    # initialize values
    initialImageSize = (len(img), len(img[0]), len(img[0][0]))
    numfeatures = len(img[0][0])
    numrows = len(img)*len(img[0])
    img = np.reshape(img, (numrows, numfeatures))
    gammas = np.zeros((k, numrows))
    gammas1 = np.zeros((k, numrows))
    gammasTimesX = [0] * k
    lamb = 0
    if flag == 1:
        lamb = 1.5E-3
    # plotting for part b
    x = []
    y = []

    # initialize class means using k-means
    kmeans = cluster.KMeans(n_clusters=k, random_state=0, max_iter=3).fit(img)
    means = kmeans.cluster_centers_

    # initialize covariance matrix for each class
    S = [np.identity(numfeatures) for i in range(k)]

    # initialize pi for each class
    PC = [1/k] * k
    for i in range(k):
        PC[i] = np.sum(kmeans.labels_ == i)/numrows

    for numiterations in range(100):

        # E step: find gamma for each pixel
        for i in range(k):
            try:
                gammas1[i] = mul(PC[i], stats.multivariate_normal.pdf(
                    x=img, mean=means[i], cov=S[i]))
            except np.linalg.LinAlgError:
                print(
                    "Singular matrix was created. Try using improved EMG by setting flag = 1.")
                return -1
        norm = np.sum(gammas1, axis=0)
        gammas = np.divide(gammas1, norm)  # normalize gammas
        # add expected complete log likelihood to y
        eps = 1E-8
        logGammas1 = np.log(gammas1+eps)
        gammasTimesLogGammas1 = mul(logGammas1, gammas)
        x = np.append(x, numiterations)
        y = np.append(y, np.sum(gammasTimesLogGammas1))

        # M step:
        # find class counts Ni
        N = np.sum(gammas, axis=1)

        # find class means
        for i in range(k):
            gammasTimesX[i] = mul(gammas[i], img.transpose())

        for i in range(k):
            means[i] = np.true_divide(np.sum(gammasTimesX[i], axis=1), N[i])

        # find pi for each class
        for i in range(k):
            PC[i] = N[i] / numrows

        for i in range(k):
            temp = img-means[i]
            S[i] = np.true_divide(np.sum(
                ((temp[:, :, None]*(temp[:, None])).swapaxes(0, 2)*gammas[i]).swapaxes(2, 0), axis=0), N[i]) + lamb*np.identity(numfeatures)

        # add expected complete log likelihood to y
        for i in range(k):
            try:
                gammas1[i] = mul(PC[i], stats.multivariate_normal.pdf(
                    x=img, mean=means[i], cov=S[i]))
            except np.linalg.LinAlgError:
                print(
                    "Singular matrix was created. Try using improved EMG by setting flag = 1.")
                return -1
        logGammas1 = np.log(gammas1+eps)
        gammasTimesLogGammas1 = mul(logGammas1, gammas)
        x = np.append(x, numiterations+.5)
        y = np.append(y, np.sum(gammasTimesLogGammas1))

    # Algorithm now finished. Now let's plot the compressed image
    compressedLabels = np.argmax(gammas, axis=0)
    compressedIMG = [means[i] for i in compressedLabels]
    compressedIMG = np.reshape(compressedIMG, initialImageSize)
    io.imshow(compressedIMG)
    title = "EM Algorithm for K = " + str(k)
    plt.title(title)
    io.show()

    # plot log likelihood
    for i in range(int(0.5*len(x)-1)):
        plt.plot(x[2*i:2*i+2], y[2*i:2*i+2], 'red')
        plt.plot(x[2*i+1:2*i+3], y[2*i+1:2*i+3], 'blue')
    plt.title("Expected Complete Log Likelihood")
    plt.xlabel("Number of EM iterations")
    plt.ylabel("E(Lc)")
    plt.show()

    # return (h,m,Q)
    print("h = ", np.transpose(gammas))
    print("m = ", means)
    print("Q = ", y)
    return (np.transpose(gammas), means, y)

# EMG("stadium.jpg", 8, 1)
