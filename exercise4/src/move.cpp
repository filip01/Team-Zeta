#include "ros/ros.h"

#include <nav_msgs/GetMap.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#include <geometry_msgs/Twist.h>
#define PI 3.141592

using namespace std;
using namespace cv;

Mat cv_map;
float map_resolution = 0;
geometry_msgs::TransformStamped map_transform;
ros::Subscriber map_sub;
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;



void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg_map) {
    int size_x = msg_map->info.width;
    int size_y = msg_map->info.height;

    if ((size_x < 3) || (size_y < 3) ) {
        ROS_INFO("Map size is only x: %d,  y: %d . Not running map to image conversion", size_x, size_y);
        return;
    }

    // resize cv image if it doesn't have the same dimensions as the map
    if ( (cv_map.rows != size_y) && (cv_map.cols != size_x)) {
        cv_map = cv::Mat(size_y, size_x, CV_8U);
    }

    map_resolution = msg_map->info.resolution;
    map_transform.transform.translation.x = msg_map->info.origin.position.x;
    map_transform.transform.translation.y = msg_map->info.origin.position.y;
    map_transform.transform.translation.z = msg_map->info.origin.position.z;

    map_transform.transform.rotation = msg_map->info.origin.orientation;

    //tf2::poseMsgToTF(msg_map->info.origin, map_transform);

    const std::vector<int8_t>& map_msg_data (msg_map->data);

    unsigned char *cv_map_data = (unsigned char*) cv_map.data;

    //We have to flip around the y axis, y for image starts at the top and y for map at the bottom
    int size_y_rev = size_y-1;

    for (int y = size_y_rev; y >= 0; --y) {

        int idx_map_y = size_x * (size_y -y);
        int idx_img_y = size_x * y;

        for (int x = 0; x < size_x; ++x) {

            int idx = idx_img_y + x;

            switch (map_msg_data[idx_map_y + x])
            {
            case -1:
                cv_map_data[idx] = 127;
                break;

            case 0:
                cv_map_data[idx] = 255;
                break;

            case 100:
                cv_map_data[idx] = 0;
                break;
            }
        }
    }

}

int points[6][2] = { 285, 198, 282, 161, 306, 185, 335, 191, 333, 209, 306, 211};
double goalOrientations[6] = { (double)7/5, (double)6/5, (double)1/4,(double)1/2, (double) 1/4, (double)1/3 };
int rotationTime[6] = {3, 5, 6, 3, 4, 6};
double rotationSpeed[6] = {-0.7, -0.7, 0.7, -0.7, -0.7, 0.7};

int current=0;



 bool goals(move_base_msgs::MoveBaseGoal goal,  ros::Publisher &pub) {
    
    if(cv_map.empty() )
        return false;

    if(current >= 6){
       ROS_INFO("Objective completed!");
      return true;
    }

    //tell the action client that we want to spin a thread by default
     MoveBaseClient ac("move_base", true);
   
     //wait for the action server to come up
     while(!ac.waitForServer(ros::Duration(5.0))){
       ROS_INFO("Waiting for the move_base action server to come up");
     }

    
    int x=points[current][0];
    int y=points[current][1];


    int v = (int)cv_map.at<unsigned char>(y,x);
    


	if (v != 255) {
		ROS_WARN("Unable to move to (x: %d, y: %d), not reachable. Continuing with the next checkpoint...", x, y);
		return false;
	}

    geometry_msgs::Point pt;
    geometry_msgs::Point transformed_pt;

    pt.x = (float)x * map_resolution;
    pt.y = (float)(cv_map.rows - y) * map_resolution;
    pt.z = 0.0;
    
    tf2::doTransform(pt, transformed_pt, map_transform);

    ROS_INFO("Moving to (x: %f, y: %f)", transformed_pt.x, transformed_pt.y);

    
     goal.target_pose.header.frame_id = "map";
     goal.target_pose.header.stamp = ros::Time::now();
     goal.target_pose.pose.position.x = transformed_pt.x;
     goal.target_pose.pose.position.y = transformed_pt.y;

     tf2::Quaternion q;
     double r=3.14159;
     double angle = goalOrientations[current];
  
     q.setRPY(0,0, angle *  r);

     goal.target_pose.pose.orientation.z = q.getZ();
     goal.target_pose.pose.orientation.w = q.getW();

     goal.target_pose.pose.orientation.x = 0;
     goal.target_pose.pose.orientation.y = 0;
     ac.sendGoal(goal);
     ac.waitForResult();
   
     if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){

       ROS_INFO("Goal reached!, spinninig...");
        ros::Time begin = ros::Time::now();
        ros::Rate rate(10);

            geometry_msgs::Twist msg;
          
           

         while(ros::Time::now() - begin < (ros::Duration) rotationTime[current]){
            msg.angular.z=rotationSpeed[current];
            pub.publish(msg);
            rate.sleep();
         }
           ROS_INFO("Done spinning");

        } else {
            ROS_INFO("Could not reach my goal, continuing with the next checkpoint");
        }
        
        current++;

       return false;
}

int numOfFaces=1;

//TO DO

void newFaceLocation(const geometry_msgs::Pose pose)
{
   /* //tell the action client that we want to spin a thread by default
     MoveBaseClient ac("move_base", true);

      //wait for the action server to come up
     while(!ac.waitForServer(ros::Duration(5.0))){
       ROS_INFO("Waiting for the move_base action server to come up");
     }
    */
    ROS_INFO("Got a new face! num: %d",numOfFaces);
    numOfFaces++;
    ROS_INFO("X: %f Y: %f", pose.position.x, pose.position.y);
    double x = (double) pose.position.x;
    double y= (double) pose.position.y-0.5;
    
    /*move_base_msgs::MoveBaseGoal goalApproach;

     goalApproach.target_pose.header.frame_id = "map";
     goalApproach.target_pose.header.stamp = ros::Time::now();
     goalApproach.target_pose.pose.position.x = x;
     goalApproach.target_pose.pose.position.y = y; 

     tf2::Quaternion q; 
     double r=3.14159;
     double angle = 0.5 ;
  
     q.setRPY(0,0, angle *  r);

     goalApproach.target_pose.pose.orientation.z = q.getZ();
     goalApproach.target_pose.pose.orientation.w = q.getW();
     goalApproach.target_pose.pose.orientation.x = 0;
     goalApproach.target_pose.pose.orientation.y = 0;
      //ROS_INFO("hey1 %f %f",q.getZ(), q.getW()); 

     ac.sendGoal(goalApproach);
     ac.waitForResult();

     if(ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED){

           ROS_INFO("Face approached");
          ros::Duration(1).sleep();

        } else {
            ROS_INFO("Could not approach face :(");
        }
  */
  
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "move");
    ros::NodeHandle n;
    

    map_sub = n.subscribe("map", 10, &mapCallback);
    move_base_msgs::MoveBaseGoal goal;

    
   ros::Publisher pub = n.advertise<geometry_msgs::Twist>("/mobile_base/commands/velocity", 1000);
   ros::Subscriber sub = n.subscribe<geometry_msgs::Pose>("new_faces",1000, newFaceLocation);
   
    while(ros::ok()) {
        
        if(goals(goal, pub)){
          return 0;
        }

        ros::spinOnce();
    }
    return 1;

}