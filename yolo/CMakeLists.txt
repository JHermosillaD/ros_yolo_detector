cmake_minimum_required(VERSION 3.0.2)
project(yolo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  custom_uv_msgs
)

###################################
## catkin specific configuration ##
###################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES yolo
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
# add_executable(${PROJECT_NAME}_node src/yolo_node.cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

## Mark executable scripts (Python etc.) 
catkin_install_python(PROGRAMS scripts/face_detection.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
