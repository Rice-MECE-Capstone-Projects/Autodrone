# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "fusion_sensor: 1 messages, 0 services")

set(MSG_I_FLAGS "-Ifusion_sensor:/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg;-Istd_msgs:/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(fusion_sensor_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_custom_target(_fusion_sensor_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "fusion_sensor" "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" "std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(fusion_sensor
  "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg"
  "${MSG_I_FLAGS}"
  "/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fusion_sensor
)

### Generating Services

### Generating Module File
_generate_module_cpp(fusion_sensor
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fusion_sensor
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(fusion_sensor_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(fusion_sensor_generate_messages fusion_sensor_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_dependencies(fusion_sensor_generate_messages_cpp _fusion_sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fusion_sensor_gencpp)
add_dependencies(fusion_sensor_gencpp fusion_sensor_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fusion_sensor_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(fusion_sensor
  "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg"
  "${MSG_I_FLAGS}"
  "/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fusion_sensor
)

### Generating Services

### Generating Module File
_generate_module_eus(fusion_sensor
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fusion_sensor
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(fusion_sensor_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(fusion_sensor_generate_messages fusion_sensor_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_dependencies(fusion_sensor_generate_messages_eus _fusion_sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fusion_sensor_geneus)
add_dependencies(fusion_sensor_geneus fusion_sensor_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fusion_sensor_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(fusion_sensor
  "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg"
  "${MSG_I_FLAGS}"
  "/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fusion_sensor
)

### Generating Services

### Generating Module File
_generate_module_lisp(fusion_sensor
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fusion_sensor
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(fusion_sensor_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(fusion_sensor_generate_messages fusion_sensor_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_dependencies(fusion_sensor_generate_messages_lisp _fusion_sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fusion_sensor_genlisp)
add_dependencies(fusion_sensor_genlisp fusion_sensor_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fusion_sensor_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(fusion_sensor
  "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg"
  "${MSG_I_FLAGS}"
  "/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fusion_sensor
)

### Generating Services

### Generating Module File
_generate_module_nodejs(fusion_sensor
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fusion_sensor
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(fusion_sensor_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(fusion_sensor_generate_messages fusion_sensor_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_dependencies(fusion_sensor_generate_messages_nodejs _fusion_sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fusion_sensor_gennodejs)
add_dependencies(fusion_sensor_gennodejs fusion_sensor_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fusion_sensor_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(fusion_sensor
  "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg"
  "${MSG_I_FLAGS}"
  "/home/zeningli/miniforge3/envs/ros_env/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fusion_sensor
)

### Generating Services

### Generating Module File
_generate_module_py(fusion_sensor
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fusion_sensor
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(fusion_sensor_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(fusion_sensor_generate_messages fusion_sensor_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zeningli/Documents/autodrone_ws/src/fusion_sensor/msg/MyArray.msg" NAME_WE)
add_dependencies(fusion_sensor_generate_messages_py _fusion_sensor_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(fusion_sensor_genpy)
add_dependencies(fusion_sensor_genpy fusion_sensor_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS fusion_sensor_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fusion_sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/fusion_sensor
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(fusion_sensor_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fusion_sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/fusion_sensor
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(fusion_sensor_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fusion_sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/fusion_sensor
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(fusion_sensor_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fusion_sensor)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/fusion_sensor
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(fusion_sensor_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fusion_sensor)
  install(CODE "execute_process(COMMAND \"/home/zeningli/miniforge3/envs/ros_env/bin/python3.11\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fusion_sensor\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/fusion_sensor
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(fusion_sensor_generate_messages_py std_msgs_generate_messages_py)
endif()
