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

# Utility rule file for camera_test_generate_messages_cpp.

# Include any custom commands dependencies for this target.
include camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/progress.make

camera_test/CMakeFiles/camera_test_generate_messages_cpp: /home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h

/home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h: /home/zeningli/miniforge3/envs/ros_env/lib/gencpp/gen_cpp.py
/home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h: /home/zeningli/Documents/autodrone_ws/src/camera_test/msg/MyArray.msg
/home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h: /home/zeningli/miniforge3/envs/ros_env/share/std_msgs/msg/Header.msg
/home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h: /home/zeningli/miniforge3/envs/ros_env/share/gencpp/msg.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/zeningli/Documents/autodrone_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating C++ code from camera_test/MyArray.msg"
	cd /home/zeningli/Documents/autodrone_ws/src/camera_test && /home/zeningli/Documents/autodrone_ws/build/catkin_generated/env_cached.sh /home/zeningli/miniforge3/envs/ros_env/bin/python3.11 /home/zeningli/miniforge3/envs/ros_env/share/gencpp/cmake/../../../lib/gencpp/gen_cpp.py /home/zeningli/Documents/autodrone_ws/src/camera_test/msg/MyArray.msg -Icamera_test:/home/zeningli/Documents/autodrone_ws/src/camera_test/msg -Istd_msgs:/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg -p camera_test -o /home/zeningli/Documents/autodrone_ws/devel/include/camera_test -e /home/zeningli/miniforge3/envs/ros_env/share/gencpp/cmake/..

camera_test_generate_messages_cpp: camera_test/CMakeFiles/camera_test_generate_messages_cpp
camera_test_generate_messages_cpp: /home/zeningli/Documents/autodrone_ws/devel/include/camera_test/MyArray.h
camera_test_generate_messages_cpp: camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/build.make
.PHONY : camera_test_generate_messages_cpp

# Rule to build all files generated by this target.
camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/build: camera_test_generate_messages_cpp
.PHONY : camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/build

camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/clean:
	cd /home/zeningli/Documents/autodrone_ws/build/camera_test && $(CMAKE_COMMAND) -P CMakeFiles/camera_test_generate_messages_cpp.dir/cmake_clean.cmake
.PHONY : camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/clean

camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/depend:
	cd /home/zeningli/Documents/autodrone_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zeningli/Documents/autodrone_ws/src /home/zeningli/Documents/autodrone_ws/src/camera_test /home/zeningli/Documents/autodrone_ws/build /home/zeningli/Documents/autodrone_ws/build/camera_test /home/zeningli/Documents/autodrone_ws/build/camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : camera_test/CMakeFiles/camera_test_generate_messages_cpp.dir/depend
