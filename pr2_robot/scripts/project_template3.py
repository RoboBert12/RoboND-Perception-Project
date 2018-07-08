#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import os
import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import copy
import matplotlib.pyplot as plt

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

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
    
    
# Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []
    detected_objects_clouds =[]
    
    # Classify the clusters! (loop through each detected cluster one at a time)

    for obj_num, cloud_indices in enumerate(cluster_indices):
        obj_temp =[]    
        for ind, pnt in enumerate(cluster_indices[obj_num]):
            
            obj_temp.append([cloud_objects[pnt][0],
                           cloud_objects[pnt][1],
                           cloud_objects[pnt][2],
                           cloud_objects[pnt][3]])
        
        good_object_cluster = pcl.PointCloud_PointXYZRGB()
        good_object_cluster.from_list(obj_temp)
        
            
    # DONE: convert the cluster from pcl to ROS using helper function
        ros_good_object_cluster = pcl_to_ros(good_object_cluster)
        detected_objects_clouds.append(ros_good_object_cluster)
    
        # Extract histogram features
        # DONE: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_good_object_cluster, using_hsv=True, nbins=85)
        normals = get_normals(ros_good_object_cluster)
        nhists = compute_normal_histograms(normals, nbins=85)
        feature = np.concatenate((chists, nhists))
    
    
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
    
        # Publish a label into RViz
        label_pos = list(white_cloud[cloud_indices[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, ind))
        
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_good_object_cluster
        detected_objects.append(do)
        
    
        
  
#       Publish a cloud for each object if it exists
#    if len(cluster_indices)>=1:
#        pcl_obj0_pub.publish(detected_objects_clouds[0])
#        
#    
#    if len(cluster_indices)>=2:
#        pcl_obj1_pub.publish(detected_objects_clouds[1])
#
#    if len(cluster_indices)>=3:
#        pcl_obj2_pub.publish(detected_objects_clouds[2])
#        
#
#    if len(cluster_indices)>=4:
#        pcl_obj3_pub.publish(detected_objects_clouds[3])
#
#    if len(cluster_indices)>=5:
#        pcl_obj4_pub.publish(detected_objects_clouds[4])
#
#    if len(cluster_indices)>=6:
#        pcl_obj5_pub.publish(detected_objects_clouds[5])
#
#    if len(cluster_indices)>=7:
#        pcl_obj6_pub.publish(detected_objects_clouds[6])
#
#    if len(cluster_indices)>=8:
#        pcl_obj7_pub.publish(detected_objects_clouds[7])
        

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    print('')
    

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # DONE: Initialize variables
    TEST_SCENE_NUM = Int32()
    OBJECT_NAME = String()
    WHICH_ARM = String()
    PICK_POSE = Pose()
    PLACE_POSE = Pose()
    
    TEST_SCENE_NUM.data = 3
    
    output_list = []
    
    # Make the dictionaries
    object_param_dict = {}
    dropbox_param_dict = {}    

    # DONE: Get/Read parameters
    # get parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_param_list = rospy.get_param('/dropbox')
    
    # DONE: Parse parameters into individual variables
    # cycle through drop box params with key [group] mapping to arm name, group 
    # and position
    for i in range(0, len(dropbox_param_list)):
        dropbox_param_dict[dropbox_param_list[i]['group']] = dropbox_param_list[i]
    
    # make dictionary for objects
    for i in range(0, len(object_list_param)):
        # Make dict object_param where key [name] maps to the [group] 
        # properties of arm name, group and position
        object_param_dict[object_list_param[i]['name']] = dropbox_param_dict[object_list_param[i]['group']]        
    
    # TODO: Rotate PR2 in place to capture side tables for the collision map (BONUS)
    
    r_drops = 0
    l_drops = 0    
    
    print('Looping through '+str(len(object_list_param))+' objects')
    print('')    
    
    # DONE: Loop through the pick list
    for object_to_pick in object_param_dict.keys():
        print('Looking for '+object_to_pick)
        
        # set flag to know that object was found        
        obj_found = False        
        
        # Cycle through the detected objects to find the one that is needed
        for idx, detected in enumerate(object_list):
            
            # See if this is the one                                  
            if detected.label == object_to_pick:
                obj_found = True
                break
        
        # decide what to do (found vs not found)
        if  obj_found:
            print(object_to_pick+' was found')
            
            # Grab name for 'pick_place_routine' service
            OBJECT_NAME.data = object_to_pick            
            
            # DONE: Assign the arm to be used for pick_place
            WHICH_ARM.data = object_param_dict[object_to_pick]['name']
            
            # DONE: Get the PointCloud for a given object and obtain it's centroid
            obj_pnt_cloud_array = ros_to_pcl(detected.cloud).to_array()
            centroid = np.mean(obj_pnt_cloud_array, axis=0)[:3]
             
            # Give the pick pose
            PICK_POSE.position.x = np.asscalar(centroid[0])
            PICK_POSE.position.y = np.asscalar(centroid[1])
            PICK_POSE.position.z = np.asscalar(centroid[2])
            PICK_POSE.orientation.x = 0.0
            PICK_POSE.orientation.y = 0.0
            PICK_POSE.orientation.z = 0.0
            PICK_POSE.orientation.w = 0.0    

            y_mod = float(0.06) # left to right
            x_mod = float(0.2) # Front to back
            
            # DONE: Create 'place_pose' for the object
            if WHICH_ARM.data == 'right':
                r_drops += 1
                drops = copy.deepcopy(r_drops)
            else:
                l_drops += 1
                drops = copy.deepcopy(l_drops)
            
            # some variation
            if np.mod(drops,2) == 1:
                y_mod = -y_mod
            
            x_mod = float((np.mod(drops,3) -1) * x_mod)

                
            drop_off_pnt = object_param_dict[object_to_pick]['position']

            PLACE_POSE.position.x = drop_off_pnt[0] + x_mod
            PLACE_POSE.position.y = drop_off_pnt[1] + y_mod
            PLACE_POSE.position.z = drop_off_pnt[2]
            PLACE_POSE.orientation.x = 0.0
            PLACE_POSE.orientation.y = 0.0
            PLACE_POSE.orientation.z = 0.0
            PLACE_POSE.orientation.w = 0.0
            
            
            # DONE: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
            # Add the object's yaml dict to the output_list
            obj_yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM,
                                           OBJECT_NAME, PICK_POSE,
                                           PLACE_POSE)
                                           
                                           
            output_list.append(obj_yaml_dict)
            
            # Wait for 'pick_place_routine' service to come up
            rospy.wait_for_service('pick_place_routine')
    
            try:
                pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
    
                # DONE: Insert your message variables to be sent as a service request
                resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)
    
                print ("Response: ",resp.success)
    
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e
             
        else:
            print(object_to_pick+' was NOT found')            
        


    # Output your request parameters into output yaml file
    send_to_yaml(os.path.expanduser('~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts'+"output_"+str(3)+".yaml"), output_list)
    send_to_yaml("output_"+str(3)+".yaml", output_list)


if __name__ == '__main__':

    # DONE: ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    
    # DONE: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # DONE: Create Publishers

    pcl_passed_pub = rospy.Publisher("/pcl_passsed", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    
#    #For each object    
#    pcl_obj0_pub = rospy.Publisher("/pcl_object0", PointCloud2, queue_size=1)
#    pcl_obj1_pub = rospy.Publisher("/pcl_object1", PointCloud2, queue_size=1)
#    pcl_obj2_pub = rospy.Publisher("/pcl_object2", PointCloud2, queue_size=1)
#    pcl_obj3_pub = rospy.Publisher("/pcl_object3", PointCloud2, queue_size=1)
#    pcl_obj4_pub = rospy.Publisher("/pcl_object4", PointCloud2, queue_size=1)
#    pcl_obj5_pub = rospy.Publisher("/pcl_object5", PointCloud2, queue_size=1)
#    pcl_obj6_pub = rospy.Publisher("/pcl_object6", PointCloud2, queue_size=1)
#    pcl_obj7_pub = rospy.Publisher("/pcl_object7", PointCloud2, queue_size=1)
    
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # DONE: Load Model From disk
#    model = pickle.load(open(os.path.expanduser('~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/pr2_model_100.sav'), 'rb'))
#    model = pickle.load(open(os.path.expanduser('~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/pr2_model_orient.sav'), 'rb'))
    model = pickle.load(open(os.path.expanduser('~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/pr2_model_60_85bins.sav'), 'rb'))
    
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # DONE: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
