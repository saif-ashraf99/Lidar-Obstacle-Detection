// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"
#include <unordered_set>

//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

	// Voxel grid point reduction
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered{new pcl::PointCloud<PointT>};
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    // Region of Interest (ROI) based filtering
    typename pcl::PointCloud<PointT>::Ptr cloudRegion{new pcl::PointCloud<PointT>};
    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloudFiltered);
    region.filter(*cloudRegion);

    // Get the indices of rooftop points
    std::vector<int> indices;
    pcl::CropBox<PointT> roof(true);
    roof.setMin(Eigen::Vector4f(-1.5, -1.7, -1, 1));
    roof.setMax(Eigen::Vector4f(2.6, 1.7, -0.4, 1));
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers{new pcl::PointIndices};
    for (int point: indices)
        inliers->indices.push_back(point);

    // Remove the rooftop indices
   
    pcl::ExtractIndices<PointT> extract;
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.setInputCloud(cloudRegion);
    extract.filter(*cloudRegion);
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;

}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
  // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
	typename pcl::PointCloud<PointT>::Ptr obstacleCloud(new pcl::PointCloud<PointT>());
  	typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());
  	for (int index : inliers->indices)
        planeCloud->points.push_back(cloud->points[index]);

    pcl::ExtractIndices<PointT> extract;
    // Extract the obstaclea
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstacleCloud);
  
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacleCloud, planeCloud);
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	pcl::PointIndices::Ptr inliers{new pcl::PointIndices()};
    pcl::ModelCoefficients::Ptr coefficients{new pcl::ModelCoefficients()};
    pcl::SACSegmentation<PointT> seg;
  
  	seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (maxIterations);
	seg.setDistanceThreshold (distanceThreshold);
    seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
		{
     std::cerr << "Could not estimate a planar model for the given dataset." <<std::endl;
        }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}


template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles

	// Create a KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    typename pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);

    for (pcl::PointIndices getIndices: clusterIndices)
    {
        typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);
        // Create a cluster with the extracted points belonging to the same object
        for (int index: getIndices.indices)
        {
            cloudCluster->points.push_back(cloud->points[index]);
        }
        cloudCluster->width = cloudCluster->size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;
        // Add the cluster to the return cluster vector
        clusters.push_back(cloudCluster);
    }
  
  
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}

template<typename PointT>
void ProcessPointClouds<PointT>::clusterHelper(int indice, typename pcl::PointCloud<PointT>::Ptr cloud, std::vector<int> &cluster, std::vector<bool> &processed, typename KdTree<PointT>::KdTree* tree, float distanceTol, int maxSize)
{
	if((processed[indice]== false) && (cluster.size() < maxSize) )
    {
        processed[indice]= true;
	    cluster.push_back(indice);

	    std::vector<int> nearest = tree->search(cloud->points[indice], distanceTol);

	    for(int id : nearest)
	    {
		    if(!processed[id])
			    clusterHelper(id, cloud, cluster, processed, tree, distanceTol, maxSize);
	    }
    }   
}

template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(typename pcl::PointCloud<PointT>::Ptr cloud, typename KdTree<PointT>::KdTree* tree, float distanceTol, int minSize, int maxSize)
{

	std::vector<std::vector<int>> clusters;
	std::vector<bool> processed(cloud->points.size(), false);

	int i=0;
	while(i < cloud->points.size())
	{
		if(processed[i])
		{
			i++;
			continue;
		}
		std::vector<int> cluster;
		clusterHelper(i, cloud, cluster, processed, tree, distanceTol,maxSize);
        if((cluster.size() >= minSize) && (cluster.size() <= maxSize))
		    clusters.push_back(cluster);
		
        i++;
	}
 
	return clusters;

}
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::euclideanCluster_clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float distanceTol, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
    typename KdTree<PointT>::KdTree* tree = new KdTree<PointT>;

    tree->insertCloud(cloud);

    std::vector<std::vector<int>> cluster_indices = euclideanCluster(cloud, tree, distanceTol,minSize,maxSize);


    for(std::vector<std::vector<int>>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++)
    {
        typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>);

        for(std::vector<int>::const_iterator j = it->begin(); j != it->end(); j++)
        {
            cloudCluster->points.push_back(cloud->points[*j]);
        }

        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}

template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::RansacPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	while(maxIterations--)
	{
		std::unordered_set<int> inliers;
		while(inliers.size() < 3)
			inliers.insert(rand() % (cloud->points.size()));

		float x1, x2, x3, y1, y2, y3, z1, z2, z3;
		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		z1 = cloud->points[*itr].z;
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;
		z2 = cloud->points[*itr].z;
		itr++;
		x3 = cloud->points[*itr].x;
		y3 = cloud->points[*itr].y;
		z3 = cloud->points[*itr].z;

		float a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1);
		float b = (x3 - x1)*(z2 - z1) - (x2 - x1)*(z3 - z1);
		float c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1);
		float d = -(a*x1 + b*y1 + c*z1);

		for(int index=0; index < cloud->points.size(); index++)
		{
			if(inliers.count(index)>0)
				continue;
			
			pcl::PointXYZI point=cloud->points[index];
			float x4 = point.x;
			float y4 = point.y;
			float z4 = point.z;

			float dis = fabs(a*x4 + b*y4 + c*z4 + d)/sqrt(a*a + b*b + c*c);

			if(dis <= distanceTol)
				inliers.insert(index);
		}
		if(inliers.size() > inliersResult.size())
			inliersResult = inliers;

	}


	return inliersResult;

}


