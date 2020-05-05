#include <iostream>
#include <ros/ros.h>
#include <ros/topic.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
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

tf2_ros::Buffer tf2_buffer;

typedef pcl::PointXYZRGB PointT;

exercise6::objs_info cylinders_info;

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
  std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;

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

  std::cerr << "PointCloud after filtering has: " << cloud_filtered2->points.size() << " data points." << std::endl;

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
  std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud(cloud_filtered2);
  extract.setIndices(inliers_plane);
  extract.setNegative(false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
  extract.filter(*cloud_plane);
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

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
  std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

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
      std::cerr << "Got color :" << color.c_str();
    }

    for (auto &c: cylinders_info.colors) {
      if (color.compare(c) == 0) { 
        return; // already found cylinder of this color
      }
    }

    std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size() << " data points." << std::endl;

    pcl::compute3DCentroid(*cloud_cylinder, centroid);
    std::cerr << "centroid of the cylindrical component: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << std::endl;

    //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PointStamped point_map;
    visualization_msgs::Marker marker;
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

    //std::cerr << tss ;

    tf2::doTransform(point_camera, point_map, tss);

    std::cerr << "point_camera: " << point_camera.point.x << " " << point_camera.point.y << " " << point_camera.point.z << std::endl;
    std::cerr << "point_map: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();

    marker.ns = "cylinder";
    marker.id = 0;

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

    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    marker.lifetime = ros::Duration();

    pubm.publish(marker);

    cylinders_info.poses.push_back(marker.pose);
    cylinders_info.colors.push_back(color);
    cylinders_info_pub.publish(cylinders_info);

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

  // For transforming between coordinate frames
  tf2_ros::TransformListener tf2_listener(tf2_buffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe("input", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubx = nh.advertise<pcl::PCLPointCloud2>("planes", 1);
  puby = nh.advertise<pcl::PCLPointCloud2>("cylinder", 1);
  pubCloud = nh.advertise<pcl::PCLPointCloud2>("processedCloud", 1);

  pubm = nh.advertise<visualization_msgs::Marker>("detected_cylinder", 1);

  color_client = nh.serviceClient<exercise6::cylinder_color>("cylinder_color");

  cylinders_info_pub = nh.advertise<exercise6::objs_info>("cylinders_info", 1);
  cylinders_info.poses =  std::vector<geometry_msgs::Pose>();
  cylinders_info.colors =  std::vector<std::string>();

  // Spin
  ros::spin();
}
