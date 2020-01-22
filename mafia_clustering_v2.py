
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from functools import reduce
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.csgraph


# from pyclustering.cluster.clique import clique

def ProcessClusters(clusters,n_samples):
    labels = -1 * np.ones(n_samples)
    for label,cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = label
    return labels

class HistogramCell:
    def __init__(self,logical_location,spatial_location,points=None,dimension=0):
        self.__logical_location = logical_location
        self.__spatial_location = spatial_location
        self.__points = points or []
        self.__visited = False
        self.dimension = dimension
    def CapturePoints(self,data,point_availability):
        for index_point in range(len(data)):
            if (point_availability[index_point] is True) and (data[index_point] > self.__spatial_location[0] and data[index_point] < self.__spatial_location[1]):
                self.__points.append(index_point)
                point_availability[index_point] = False
    @property
    def density(self):
        return len(self.__points)
    @property
    def logical_location(self):
        return self.__logical_location

    def Dense(self,densityThreshold):
        if len(self.__points) > densityThreshold:
            return True
        return False

def Neighbor(candidate1, candidate2):
    if candidate1.dimension == candidate2.dimension:
        print
        if np.abs(candidate1.logical_location-candidate2.logical_location) < 1:
            return 1
    return 0

def MafiaClustering(data,initialBinSize=20,processors=None,density_threshold=1,threshold=0.20):
    """n_records
    n_processors
    #dimensionality
    # memory buffer for each process.

    #Create a Histogram in each dimension. Can be multithreaded.
    #Create Global Histogram and calculate adaptive intervals
    #Put candidate units into the bins

    #while no dense units left
    # if k > 1
    #Build candidate dense units in k from dense units in k-1
    # Read data and populate dense units
    # Pick dense units and check if dense unit
    #Pick the appropriate dense units, find bounds and build data structure for k+1 dimension
    """

    nDims = data.shape[1]

    min_corner = []
    max_corner = []
    data_sizes = []
    for curDim in range(data.shape[1]):
        mn = min(data[:,curDim])
        mx = max(data[:,curDim])
        min_corner.append(mn)
        max_corner.append(mx)
        data_sizes.append(mx - mn)

    #Performing Initial Binning of samples
    cellDim=[]
    cell_sizes = [dimension_length / initialBinSize for dimension_length in data_sizes]
    for dim in range(nDims):
        #Creating Histogram
        point_availability = [True] * len(data)
        hist =[]
        for i in range(initialBinSize):
            spatial_location = [cell_sizes[dim]*i,cell_sizes[dim]*(i+1)]
            dimData=[sample[dim] for sample in data]
            tmp = HistogramCell(i,spatial_location)
            tmp.CapturePoints(dimData,point_availability)
            hist.append(tmp.density)

        #Combine hist bins if density is within threshold% of eachother
        histBounds =[]
        idx1=0
        idx2=1
        while idx2 != initialBinSize:
            if hist[idx2] > hist[idx1]*(1-threshold)  and hist[idx2] < hist[idx1]*(1+threshold):
                idx2+=1
            else:
                histBounds.append([cell_sizes[dim]*idx1,cell_sizes[dim]*idx2])
                idx1=idx2
                idx2+=1
        histBounds.append([cell_sizes[dim]*idx1,cell_sizes[dim]*idx2])
        cellDim.append(histBounds)

    #Rebinning the data into the new cells
    candidates =[]
    for dim in range(nDims):
        #Creating Histogram
        point_availability = [True] * len(data)
        for i in range(len(cellDim[dim])):
            spatial_location = cellDim[dim][i]
            dimData=[sample[dim] for sample in data]
            tmp = HistogramCell(i,spatial_location,dimension=dim)
            tmp.CapturePoints(dimData,point_availability)
            if tmp.density > density_threshold:
                candidates.append(tmp)
    print(candidates)
    for dim in range(nDims):
        print(dim)
        if dim == 0:
            graph = np.identity(len(candidates))
            for i in range(len(candidates)):
                for j in range(len(candidates)):
                    graph[i,j] = Neighbor(candidates[i], candidates[j])
            nbConnectedComponents, components = scipy.sparse.csgraph.connected_components(
                graph, directed=False)
            clusterAssignment = -1 * np.ones(data.shape[0])
            candidates = np.array(candidates)
            print(len(candidates))
            for i in range(nbConnectedComponents):
                # Get dense units of the cluster
                cluster_dense_units = candidates[np.where(components == i)[0]]
                # print(cluster_dense_units)
                clusterDimensions = {}
                for j in range(len(cluster_dense_units)):
                    for k in range(len(cluster_dense_units[j])):
                        if cluster_dense_units[j][k].dimension not in clusterDimensions:
                            clusterDimensions[cluster_dense_units[j][k].dimension] = []
                        clusterDimensions[cluster_dense_units[j][k].dimension].extend(cluster_dense_units[j][k].points)
                points =reduce(np.intersect1d, list(clusterDimensions.values()))
                clusterAssignment[points] = i

            print(clusterDimensions.keys(), points)
    else:
        pass


    return 0


#Debuging and Testing
if __name__ == "__main__":
    #Generate a list of vectors
    n_processors = 1
    n_components = 10
    n_samples=5000
    data, truth = make_blobs(n_samples=n_samples, centers=n_components, random_state=43, n_features=5)
    data = preprocessing.MinMaxScaler().fit_transform(data)

    intervals = 14  # defines amount of cells in grid in each dimension
    density_threshold = 25   # lets consider each point as non-outlier
    clusters = MafiaClustering(data, 14, density_threshold=density_threshold, threshold=0.3)
    clusters = MafiaClustering(data, 9, density_threshold=density_threshold, threshold=0.3)
    clusters = MafiaClustering(data, 4, density_threshold=density_threshold, threshold=0.3)

    # print(labels)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # ax1.scatter(data[:, 0], data[:, 1],data[:, 2],  c=labels, marker='o')
    # ax1.set_xlabel("Feature 1")
    # ax1.set_ylabel("Feature 2")
    # ax1.set_zlabel("Feature 3")
    # plt.show()
