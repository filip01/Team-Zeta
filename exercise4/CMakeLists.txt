cmake_minimum_required(VERSION 2.8.3)
project(exercise4)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	cv_bridge
	sensor_msgs
	visualization_msgs
	tf2_ros
	nav_msgs
	std_msgs
	geometry_msgs
	sound_play
	message_generation
)



add_message_files(
  FILES
  faces_info.msg
)

generate_messages(
	DEPENDENCIES
	roscpp
	sensor_msgs
	visualization_msgs
	std_msgs
	geometry_msgs	
)
catkin_package(
	CATKIN_DEPENDS 
	message_runtime 
)
find_package( OpenCV REQUIRED )

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_executable(move src/move.cpp)
target_link_libraries(move ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

#catkin_install_python(PROGRAMS scripts/detect_rings DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

