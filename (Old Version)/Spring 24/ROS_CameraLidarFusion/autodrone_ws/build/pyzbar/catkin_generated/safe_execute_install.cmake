execute_process(COMMAND "/home/zeningli/Documents/autodrone_ws/build/pyzbar/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/zeningli/Documents/autodrone_ws/build/pyzbar/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
