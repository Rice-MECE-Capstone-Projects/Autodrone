# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/zeningli/miniforge3/envs/ros_env/bin/cmake

# The command to remove a file.
RM = /home/zeningli/miniforge3/envs/ros_env/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zeningli/Documents/autodrone_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zeningli/Documents/autodrone_ws/build

# Include any dependencies generated for this target.
include rplidar_ros/CMakeFiles/rplidarNodeClient.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include rplidar_ros/CMakeFiles/rplidarNodeClient.dir/compiler_depend.make

# Include the progress variables for this target.
include rplidar_ros/CMakeFiles/rplidarNodeClient.dir/progress.make

# Include the compile flags for this target's objects.
include rplidar_ros/CMakeFiles/rplidarNodeClient.dir/flags.make

rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o: rplidar_ros/CMakeFiles/rplidarNodeClient.dir/flags.make
rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o: /home/zeningli/Documents/autodrone_ws/src/rplidar_ros/src/client.cpp
rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o: rplidar_ros/CMakeFiles/rplidarNodeClient.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/zeningli/Documents/autodrone_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o"
	cd /home/zeningli/Documents/autodrone_ws/build/rplidar_ros && /home/zeningli/miniforge3/envs/ros_env/bin/aarch64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o -MF CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o.d -o CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o -c /home/zeningli/Documents/autodrone_ws/src/rplidar_ros/src/client.cpp

rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/rplidarNodeClient.dir/src/client.cpp.i"
	cd /home/zeningli/Documents/autodrone_ws/build/rplidar_ros && /home/zeningli/miniforge3/envs/ros_env/bin/aarch64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zeningli/Documents/autodrone_ws/src/rplidar_ros/src/client.cpp > CMakeFiles/rplidarNodeClient.dir/src/client.cpp.i

rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/rplidarNodeClient.dir/src/client.cpp.s"
	cd /home/zeningli/Documents/autodrone_ws/build/rplidar_ros && /home/zeningli/miniforge3/envs/ros_env/bin/aarch64-conda-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zeningli/Documents/autodrone_ws/src/rplidar_ros/src/client.cpp -o CMakeFiles/rplidarNodeClient.dir/src/client.cpp.s

# Object files for target rplidarNodeClient
rplidarNodeClient_OBJECTS = \
"CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o"

# External object files for target rplidarNodeClient
rplidarNodeClient_EXTERNAL_OBJECTS =

/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: rplidar_ros/CMakeFiles/rplidarNodeClient.dir/src/client.cpp.o
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: rplidar_ros/CMakeFiles/rplidarNodeClient.dir/build.make
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libroscpp.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_chrono.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_filesystem.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libxmlrpcpp.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/librosconsole.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/librosconsole_log4cxx.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/librosconsole_backend_interface.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/liblog4cxx.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_regex.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libroscpp_serialization.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/librostime.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_date_time.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libcpp_common.so
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_system.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libboost_thread.so.1.82.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: /home/zeningli/miniforge3/envs/ros_env/lib/libconsole_bridge.so.1.0
/home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient: rplidar_ros/CMakeFiles/rplidarNodeClient.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/zeningli/Documents/autodrone_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient"
	cd /home/zeningli/Documents/autodrone_ws/build/rplidar_ros && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rplidarNodeClient.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
rplidar_ros/CMakeFiles/rplidarNodeClient.dir/build: /home/zeningli/Documents/autodrone_ws/devel/lib/rplidar_ros/rplidarNodeClient
.PHONY : rplidar_ros/CMakeFiles/rplidarNodeClient.dir/build

rplidar_ros/CMakeFiles/rplidarNodeClient.dir/clean:
	cd /home/zeningli/Documents/autodrone_ws/build/rplidar_ros && $(CMAKE_COMMAND) -P CMakeFiles/rplidarNodeClient.dir/cmake_clean.cmake
.PHONY : rplidar_ros/CMakeFiles/rplidarNodeClient.dir/clean

rplidar_ros/CMakeFiles/rplidarNodeClient.dir/depend:
	cd /home/zeningli/Documents/autodrone_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zeningli/Documents/autodrone_ws/src /home/zeningli/Documents/autodrone_ws/src/rplidar_ros /home/zeningli/Documents/autodrone_ws/build /home/zeningli/Documents/autodrone_ws/build/rplidar_ros /home/zeningli/Documents/autodrone_ws/build/rplidar_ros/CMakeFiles/rplidarNodeClient.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : rplidar_ros/CMakeFiles/rplidarNodeClient.dir/depend
