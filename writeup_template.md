## Project: Perception Pick & Place

# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
For this project I used the script project_template3.py.

I started the code by changing the ros message into a point cloud. To deal with the noise, I implemented a statistical outlier filter. I considered the 20 points around each point and set the scale factor to 0.6. This filter helped to remove most of the noise in the signal.
```python
# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # DONE: Convert ROS msg to PCL data
    plc_cloud =  ros_to_pcl(pcl_msg)

    # DONE: Statistical Outlier Filtering
    # Much like the previous filters, we start by creating a filter object: 
    outlier_filter = plc_cloud.make_statistical_outlier_filter()
    
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(20)
    
    # Set threshold scale factor
    x = .6
    
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    
    # Finally call the filter function for magic
    outlier_filtered = outlier_filter.filter()    
```

Next, I down sampled by cloud to make it eaiser to work with. Initally I had ok results with a 0.01 leaf size. However, I had trouble identifying the book and glue. To address that issue I chose to make my leaf size smaller, allowing a richer cloud to be processed. After halving the number twice, I was able to identify the book. It should be noted that for other environments, the leaf size that I used is excessive, but I wanted to be able to use one script for all three environments.

```python
    # DONE: Voxel Grid Downsampling
    # Choose a voxel (also known as leaf) size
    # LOWER NUMBER = HIGH DENSITY
    LEAF_SIZE = .0025

    # Create a VoxelGrid filter object for our input point cloud
    vox = outlier_filtered.make_voxel_grid_filter()    
    
    # Set the voxel (or leaf) size 
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    
    # Call the filter function to obtain the resultant downsampled point cloud
    vox_filtered = vox.filter()
```

After down sampling, I needed to filter out the area that the objects apeared in. I did this using a pass through filter. The first was a Z axis filter similar to what was used in the exercises. I did need to increase the upper limit, to avoid cutting off the tops of taller objects. After filtering the cloud, published a ros msg with the results. By visualizing this cloud, I saw that the edges of the bins were seen by the robot. This meant that during clustering I would see 2 extra objects. I added a secdond filter in the y axis to eliminate the bins. 

```python
    # DONE: PassThrough Filter
    # Create a PassThrough filter object.
    passthrough = vox_filtered.make_passthrough_filter()
    
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.25
    passthrough.set_filter_limits(axis_min, axis_max)

    # Apply passthrough filter
    cloud_passed = passthrough.filter()
    
    # also filiter in y axis to get rid of bin corners
    cloud_passed_z = cloud_passed.make_passthrough_filter()
    
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    cloud_passed_z.set_filter_field_name(filter_axis)
    axis_min = -0.4 
    axis_max = 0.4
    cloud_passed_z.set_filter_limits(axis_min, axis_max)

    # Apply passthrough filter
    cloud_passed = cloud_passed_z.filter()
    
    cloud_passed_4msg = copy.deepcopy(cloud_passed)
```
Now I needed to remove the table using a RANSAC plane. I implemented the version of this that I used in exercise 1. After I found the plane. I grabbed the indexes of the inliers(table) and outliers(objects). I also created clouds of these objects that were later published.

```python
    # DONE: RANSAC Plane Segmentation
    seg = cloud_passed.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    
    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    
    # DONE: Extract inliers and outliers
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()
    
    # Extract inliers and outliers
    # At this point the clouds are different
    cloud_objects = cloud_passed.extract(inliers, negative=True)
    cloud_table = cloud_passed.extract(inliers, negative=False)
```

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  
Now that the cloud containing all of the objects was found, I needed to segment them. To do this I applied Euclidean clustering. This clustering was prefered because you don't need to know how many clusters there are ahead of time. It took some trial and error to determine the min and max cluster sizes. These nubers also needed to be revise when the leaf size changed. Eventualy I settle on a tolerance of 0.2, with cluster ranging from 500 to 7000 points. This allowed me to capture the correct number of objects in all scenes. After finding all of the obejects, I looped through to build an array of clouds that represent each found cluster with a unique color. I published this cloud and the others at this step as well.

```python
    # DONE: Euclidean Clustering
    # Define max_distance (eps parameter in DBSCAN())
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(500)
    ec.set_MaxClusterSize(7000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # DONE: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    
    # container for the colored clouds
    color_cluster_point_list = []
    
    for j, indices in enumerate(cluster_indices):
        
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])
    
    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    

    # DONE: Convert PCL data to ROS messages
    # Convert PCL data to ROS messages
    ros_passed_cloud = pcl_to_ros(cloud_passed_4msg)
    ros_cloud_objects =  pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    

    # DONE: Publish ROS messages
    # Publish ROS messages
    pcl_passed_pub.publish(ros_passed_cloud)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
```


#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Now that the objects were found, I needed to clasify them. To do this I needed to first get the features of the objects and train a SVM. To obtain the object features, I used the sensor_stick model. I copied the script that was used to get the features in exercise 3 and modified it. I called the new script capture_features_pr2_100.py because I was originally going to capture 100 clouds per image. I later revised that to 50. I also modified the compute_color_histograms and compute_normal_histograms from the features.py script. I added the ability to input the number of bins in the function to make the scripts more versitle. After some trial and error, I ended up using 85 bins, that allwed for 3 full values of colors per bin (255/3 = 85). The capture_features_pr2_100 would take the histograms of colors and normals and save them to a file for training the SVM later. I went through several iterations in order to get a set that gave me accurate results. I tried to balance the number of bins to give enough seperation, but to not be too broad that features were unrecognazable. The capture_features_pr2_100 script was run after the training model was launched. In one terminal I executed the following command:

```python
roslaunch sensor_stick training.launch
```

Then in a second terminal I exectued this command to capture the features:
```python
rosrun pr2_robot capture_features_pr2_100.py
```

After geting the inital file, I needed to train it. I used a training file that I developed for exercise 3, modifing it slightly. I used a linear kernal and was able to achive good results after setting the histogram bins to 85.

![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)


```python
rosrun pr2_robot train_svm_pr2.py 
```





```python
roslaunch pr2_robot pick_place_project.launch
```

```python
rosrun pr2_robot project_template3.py
```

Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



