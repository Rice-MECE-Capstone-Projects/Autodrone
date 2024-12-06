# Install script for directory: /home/zeningli/Documents/autodrone_ws/src/lidar_3d_test

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zeningli/Documents/autodrone_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/zeningli/miniforge3/envs/ros_env/bin/aarch64-conda-linux-gnu-objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_3d_test.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/lidar_3d_test/cmake" TYPE FILE FILES
    "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_3d_testConfig.cmake"
    "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_3d_testConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/lidar_3d_test" TYPE FILE FILES "/home/zeningli/Documents/autodrone_ws/src/lidar_3d_test/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/fusion_demo.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_scan_process.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_test.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/fake_altitude.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_scan_sphere.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/lidar_scan_sphere_render.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/lidar_3d_test" TYPE PROGRAM FILES "/home/zeningli/Documents/autodrone_ws/build/lidar_3d_test/catkin_generated/installspace/camera_pub.py")
endif()

