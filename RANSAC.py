import os
import numpy as np
import matplotlib.pyplot as plt
import math

def split_at_char(s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_average_dist_to_origin(points):
    dist = (points - [0,0])**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    
    return np.mean(dist)

def read_data(folder, file):
    """
    Returns a list of lists containing the points of the 
    reference and warped images.
    """
    
    # print("Reading original and warped image datapoints...")
    
    with open(os.path.join("boat", "homography.txt"), "r") as file:
        image_data = [line for line in file]

    image1 = image_data[0]
    image1 = split_at_char(image1, "[", "]").split(";")
    image1 = [elem.split(",") for elem in image1]
    image1 = [list(map(int,i)) for i in image1]
    
    image2 = image_data[1]
    image2 = split_at_char(image2, "[", "]").split(";")
    image2 = [elem.split(",") for elem in image2]
    image2 = [list(map(int,i)) for i in image2]
    
    return image1, image2

def normalize_image_points(image):
    """
    Input: 2D list with x,y image points
    Output: 
    """
    
    image = np.array(image)
    mean, std = np.mean(image, 0), np.std(image)
    
    # define similarity transformation (no rotation, scaling using sdv and setting centroid as origin)
    Transformation = np.array([[std/np.sqrt(2), 0, mean[0]],
                               [0, std/np.sqrt(2), mean[1]],
                               [0,   0, 1]])
    
    # apply transformation on data points
    Transformation = np.linalg.inv(Transformation)
    image = np.dot(Transformation, np.concatenate((image.T, np.ones((1, image.shape[0])))))
    
    # retrieve normalized image in the original input shape (25, 2)
    image = image[0:2].T
    
    return image, Transformation

def compute_matrix_A(points1, points2, no_points):
    """
    Input: Normalized correspondences for image1 and image2
    Output: Matrix A as defined in Zisserman p. 91
    """
    
    A = []

    for i in range(0, no_points):
        x, y = points1[i, 0], points1[i, 1]
        x_prime, y_prime = points2[i, 0], points2[i, 1]
        
        # create A_i according to the eq. in the book
        # here we are assuming w_i is one
        A.append([0, 0, 0, -x, -y, -1, y_prime*x, y_prime*y, y_prime])
        A.append([x, y, 1, 0, 0, 0, -x_prime*x, -x_prime*y, -x_prime])
    
    return np.asarray(A)

def compute_SVD(matrix_A):
    """
    """
    return np.linalg.svd(matrix_A)

def get_vector_h(matrix_V):
    """
    Input: Matrix V from SVD of A
    Output: Unitary vector h (last column of V matrix of SVD)
    """
        
    h = matrix_V[-1,:]/matrix_V[-1,-1]
    
    return h

def compute_homography(datapoints1, datapoints2, sample_size):
    """
    Normalized DLT implementation of homography.
    """
    
    no_points = datapoints1.shape[0]

    # normalize data
    datapoints1_normalized, T = normalize_image_points(datapoints1)
    datapoints2_normalized, T_prime = normalize_image_points(datapoints2)

    # get matrix A for each normalized correspondence (dims 2*n x 9)
    A = compute_matrix_A(datapoints1_normalized, datapoints2_normalized, no_points)

    # compute SVD of A
    U, S, V = compute_SVD(A)

    # get last column of V and normalize it (this is the vector h)
    h = get_vector_h(V)

    H_tilde = h.reshape(3,3)

    # denormalize to obtain homography (H) using the transformations and generalized pseudo-inverse
    H = np.dot(np.dot(np.linalg.pinv(T_prime), H_tilde), T)
    
    return H

def get_euclidean_distance(point1, point2):
    """
    Standard euclidean distance for n-dimensional vectors.
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

def get_threshold_distance(datapoints):
    """
    Computes a distance threshold based on Table 4.2 of Zisserman et al.
    This is the distance for a probability of 0.95 for a point of being an inlier.
    """
    
    sigma = np.std(datapoints)
    
    return np.sqrt(5.99)*sigma

def get_transfer_distances(datapoints1, datapoints2, homography):
    """
    Computes the symmetric transfer error for each point correspondence and its projection.
    """
    
    def get_transfer_distance(pointa, pointb, homography):
        """
        Implements symmetric transfer error.
        """
        return np.sqrt(get_euclidean_distance(pointa, np.dot(np.linalg.inv(homography), pointb))**2 + get_euclidean_distance(pointb, np.dot(homography, pointa))**2)
    
    distances = list([])
    
    for point1, point2 in zip(datapoints1, datapoints2):
        
        point1_homogeneous = np.append(point1, 1)
        point2_homogeneous = np.append(point2, 1)

        transfer_distance = get_transfer_distance(point1_homogeneous, point2_homogeneous, homography)
        
        distances.append(transfer_distance)
    
    return distances

def get_inliers(distances, threshold):
    inlier_index = []
    for index in range(0, len(distances)):
        if distances[index] < threshold:
            inlier_index.append(index)
    return inlier_index

def is_collinear(points, image_points):
    m1 = (image_points[points[1]][1] - image_points[points[0]][1])/(image_points[points[1]][0] - image_points[points[0]][0])
    m2 = (image_points[points[3]][1] - image_points[points[2]][1])/(image_points[points[3]][0] - image_points[points[2]][0])
    if m1 == m2:
        return True
    else:
        return False

def RANSAC():
    """
    RANSAC algorithm.
    """

    # read image data points
    image1, image2 = read_data("boat", "homography.txt")

    image1 = np.array(image1)
    image2 = np.array(image2)
    
    # initialize parameters
    T = -1 # threshold max no. of inliers
    p = 0.99 # choose prob to ensure that at least one of the random samples of s points has no outliers
    N = np.inf
    sample_points = 4 # (minimum required to compute homography)
    best_inlier_index = []
    sample_count = 0
    min_std = 10e5
    threshold_distance = get_threshold_distance(image1)
    H = np.zeros((3,3))
    
    while (N > sample_count):
        # choose randomly 4 correspondences
        index = np.random.choice(image1.shape[0], sample_points, replace=False)
        
        # check collinearity
        if not is_collinear(index, image1):
            # sample 4 points from images
            sampled_image1 = image1[index]
            sampled_image2 = image2[index]

        # instantiate (normalized) homography with set of 4 point pairs
        H_curr = compute_homography(sampled_image1, sampled_image2, sample_points)

        # compute distances for all points with current H
        transfer_distances = get_transfer_distances(image1, image2, H_curr)
        
        # determine the set of inliers consistent with H (correspondences for which d < t)
        inliers = get_inliers(transfer_distances, threshold_distance)
        
        # compute std of inlier distances
        std_distances = np.std(transfer_distances)
        
        # count no. of inliers whose distance d < t
        no_inliers = len(inliers)
        
        # update based on best homography and keep inlier index
        if no_inliers > T or (no_inliers == T and std_distances < min_std):
            T = no_inliers
            min_std = std_distances
            best_inlier_index = inliers
            H = H_curr
        
        # update N (determine samples adaptively)
        epsilon = 1 - (len(inliers)/len(image1))
        N = np.log(1-p)/np.log(1-(1-epsilon)**sample_points)
        sample_count += 1
        
        # refine H using the inliers 
        H = compute_homography(image1[inliers], image2[inliers], best_inlier_index)
    
    RANSAC_homography = H # compute_homography(image1, image2, best_inlier_index)
    print("RANSAC homography...")
    print(RANSAC_homography)
    
    return RANSAC_homography

if __name__ == "__main__":
	RANSAC()