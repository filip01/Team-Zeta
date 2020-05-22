#include <iostream>
#include <ros/ros.h>
#include <ros/topic.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl/point_cloud.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "geometry_msgs/PointStamped.h"
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include "exercise6/cylinder_color.h"
#include "exercise6/objs_info.h"

ros::Publisher pubx;
ros::Publisher puby;
ros::Publisher pubm;
ros::Publisher pubCloud;
ros::Publisher pub_cylinder_img;
ros::Publisher cylinders_info_pub;
ros::ServiceClient color_client;
visualization_msgs::MarkerArray markerArray;
int id=0;

tf2_ros::Buffer tf2_buffer;

typedef pcl::PointXYZRGB PointT;

exercise6::objs_info cylinders_info;

geometry_msgs::PointStamped get_robot_ps(ros::Time time_rec, ros::Time time_test) {
    geometry_msgs::PointStamped point_map_robot;
    geometry_msgs::PointStamped point_base;
    geometry_msgs::TransformStamped t_base_map;

    point_base.header.frame_id = "base_link";
    point_base.header.stamp = ros::Time::now();

    point_map_robot.header.frame_id = "map";
    point_map_robot.header.stamp = ros::Time::now();

    point_base.point.x = 0;
    point_base.point.y = 0;
    point_base.point.z = 0;

    try
    {
      time_test = ros::Time::now();

      std::cerr << time_test << std::endl;
      t_base_map = tf2_buffer.lookupTransform("map", "base_link", time_rec);
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("Transform warning: %s\n", ex.what());
    }

    tf2::doTransform(point_base, point_map_robot, t_base_map);
    std::cerr << "point_base: " << point_base.point.x << " " << point_base.point.y << " " << point_base.point.z << std::endl;

    return point_map_robot;
}

void cloud_cb(const pcl::PCLPointCloud2ConstPtr &cloud_blob)
{
  // All the objects needed

  ros::Time time_rec, time_test;
  time_rec = ros::Time::now();

  pcl::PassThrough<PointT> pass;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
  Eigen::Vector4f centroid;

  // Datasets
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

  pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);

  pcl::PointCloud<PointT>::Ptr cloud_filtered3(new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals3(new pcl::PointCloud<pcl::Normal>);

  pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);

  boost::shared_ptr<geometry_msgs::Twist const> vel;
  ros::Duration timeout = (ros::Duration)0.5;

  vel = ros::topic::waitForMessage<geometry_msgs::Twist>("/mobile_base/commands/velocity", timeout);

  if (vel != NULL)
  {
    // vel = *vel1;

    if (abs(vel->linear.x) > 0.3 || abs(vel->angular.z) > 0.8)
    {
      std::cerr << "Robot velocity threshold reached! [" << vel->linear.x << ", " << vel->angular.z << "]" << std::endl;
      return;
    }
    //std::cerr << "READ VELOCITY!!";
  }

  pcl::PCLPointCloud2::Ptr cloud_downsampled(new pcl::PCLPointCloud2());

  //DOWNSAMPLE THE CLOUD!!
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud_blob);
  sor.setLeafSize(0.01f, 0.01f, 0.01f);
  sor.filter(*cloud_downsampled);

  // Read in the cloud data
  pcl::fromPCLPointCloud2(*cloud_downsampled, *cloud);
  //std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;

  // filter cloud: remove points too high
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("y");
  pass.setFilterLimits(0, 1.5);
  pass.filter(*cloud_filtered);

  // filter cloud: remove points too far away
  pass.setInputCloud(cloud_filtered);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(0, 1.7);
  pass.filter(*cloud_filtered2);

  //std::cerr << "PointCloud after filtering has: " << cloud_filtered2->points.size() << " data points." << std::endl;

  // Estimate point normals
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud_filtered2);
  ne.setKSearch(50);
  ne.compute(*cloud_normals2);

  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight(0.1);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(0.04);
  seg.setInputCloud(cloud_filtered2);
  seg.setInputNormals(cloud_normals2);

  // Obtain the plane inliers and coefficients
  seg.segment(*inliers_plane, *coefficients_plane);
  //std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud(cloud_filtered2);
  extract.setIndices(inliers_plane);
  extract.setNegative(false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
  extract.filter(*cloud_plane);
  //std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

  //publish the extracted plane
  pcl::PCLPointCloud2 outcloud_plane;
  pcl::toPCLPointCloud2(*cloud_plane, outcloud_plane);
  pubx.publish(outcloud_plane);

  // Remove the planar inliers, extract the rest
  extract.setNegative(true);
  extract.filter(*cloud_filtered2);
  extract_normals.setNegative(true);
  extract_normals.setInputCloud(cloud_normals2);
  extract_normals.setIndices(inliers_plane);
  extract_normals.filter(*cloud_normals2);

  //publish filtered cloud
  pcl::PCLPointCloud2 outcloud_filtered2;
  pcl::toPCLPointCloud2(*cloud_filtered2, outcloud_filtered2);
  pubCloud.publish(outcloud_filtered2);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(0.01);
  seg.setRadiusLimits(0.11, 0.12);
  seg.setInputCloud(cloud_filtered2);
  seg.setInputNormals(cloud_normals2);

  // Obtain the cylinder inliers and coefficients
  seg.segment(*inliers_cylinder, *coefficients_cylinder);
 // std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  extract.setInputCloud(cloud_filtered2);
  extract.setIndices(inliers_cylinder);
  extract.setNegative(false);
  pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
  extract.filter(*cloud_cylinder);

  if (cloud_cylinder->points.empty() || cloud_cylinder->points.size() < 700)
    if (!cloud_cylinder->points.empty())
    {
      std::cerr << "Cylinder detected didnt have enough points! [" << cloud_cylinder->points.size() << " ]" << std::endl;
    }
    else
    {
      std::cerr << "Can't find the cylindrical component." << std::endl;
    }
  else
  {
    
    sensor_msgs::Image image_;
    pcl::toROSMsg(*cloud_cylinder, image_); //convert the cloud

    exercise6::cylinder_color srv;
    srv.request.image_1d = image_;
    std::string color;
    if (color_client.call(srv)) {
      color = srv.response.color;
      std::cerr << "Got color :" << color.c_str() << std::endl;
    } else {
      // didnt get a color D:
      return;
    }
    float margin = 0.7;
    pcl::compute3DCentroid(*cloud_cylinder, centroid);

    /// TRANSFORM ///

    //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PointStamped point_map;
    geometry_msgs::TransformStamped tss;

    point_camera.header.frame_id = "camera_rgb_optical_frame";
    point_camera.header.stamp = ros::Time::now();

    point_map.header.frame_id = "map";
    point_map.header.stamp = ros::Time::now();

    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

    try
    {
      time_test = ros::Time::now();

      std::cerr << time_rec << std::endl;
      std::cerr << time_test << std::endl;
      tss = tf2_buffer.lookupTransform("map", "camera_rgb_optical_frame", time_rec);
      //tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN("Transform warning: %s\n", ex.what());
    }

    tf2::doTransform(point_camera, point_map, tss);

    // std::cerr << "point_camera: " << point_camera.point.x << " " << point_camera.point.y << " " << point_camera.point.z << std::endl;
    // std::cerr << "point_map: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;

    /// TRANSFORM END///

    for (auto &p: cylinders_info.poses) {
      //std::cerr << "Pose: " << p.position.x << " " << p.position.y << " " << p.position.z << std::endl;
      //std::cerr << "centroid of the cylindrical component: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;
      
      if ( point_map.point.x + margin > p.position.x  && point_map.point.x - margin < p.position.x  && point_map.point.y + margin > p.position.y && point_map.point.y - margin < p.position.y && point_map.point.z + margin > p.position.z && point_map.point.z - margin < p.position.z){
          std::cerr << "Cylinder already found!" << std::endl;
          return;
      } 
      
    }
    /* 
    
    CHECKING BASED ON COLOR, NOT NECESSARY

    for (auto &c: cylinders_info.colors) {
      if (color.compare(c) == 0) { 
        return; // already found cylinder of this color
      }
    } 
    
    */
  
    //std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size() << " data points." << std::endl;
    //std::cerr << "centroid of the cylindrical component: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << std::endl;
    


    cylinders_info.robot_point_stamped = get_robot_ps(time_rec, time_test);

    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    marker.ns = "cylinder";
    marker.id = id;
    id++;

    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = point_map.point.x;
    marker.pose.position.y = point_map.point.y;
    marker.pose.position.z = point_map.point.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    std::cerr << "COLORR:::  " << color << " " << color.compare("green") << std::endl;
    
    if(color.compare("red") == 0) {
       r = 1.0f;
       g = 0.0f;
       b = 0.0f;
    }
    if(color.compare("green") == 0) {
       r = 0.0f;
       g = 1.0f;
       b = 0.0f;
    }
    if(color.compare("blue") == 0) {
       r = 0.0f;
       g = 0.0f;
       b = 1.0f;
    }
    if(color.compare("yellow") == 0) {
       r = 1.0f;
       g = 1.0f;
       b = 0.0f;
    }
    if(color.compare("black") == 0) {
       r = 0.0f;
       g = 0.0f;
       b = 0.0f;
    }

    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0f;

    marker.lifetime = ros::Duration();

    markerArray.markers.push_back(marker);
  
    pubm.publish(markerArray);

    cylinders_info.poses.push_back(marker.pose);
    cylinders_info.colors.push_back(color);

    pcl::PCLPointCloud2 outcloud_cylinder;
    pcl::toPCLPointCloud2(*cloud_cylinder, outcloud_cylinder);
    puby.publish(outcloud_cylinder);
  }
}

int main(int argc, char **argv)
{
  // Initialize ROS
  ros::init(argc, argv, "cylinder_segment");
  ros::NodeHandle nh;
  ROS_INFO("Node started!");
  markerArray.markers = {};

  // For transforming between coordinate frames
  tf2_ros::TransformListener tf2_listener(tf2_buffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("input", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubx = nh.advertise<pcl::PCLPointCloud2>("planes", 1);
  puby = nh.advertise<pcl::PCLPointCloud2>("cylinder", 1);
  pubCloud = nh.advertise<pcl::PCLPointCloud2>("processedCloud", 1);

  pubm = nh.advertise<visualization_msgs::MarkerArray>("detected_cylinders", 1);

  color_client = nh.serviceClient<exercise6::cylinder_color>("cylinder_color");

  cylinders_info_pub = nh.advertise<exercise6::objs_info>("cylinders_info", 10);
  cylinders_info.poses =  std::vector<geometry_msgs::Pose>();
  cylinders_info.colors =  std::vector<std::string>();
  cylinders_info.robot_point_stamped = geometry_msgs::PointStamped();

  // Spin
  ros::Rate rate(10);
  while (ros::ok()) {

    
    cylinders_info_pub.publish(cylinders_info);
    ros::spinOnce();
    rate.sleep();
  }
}
