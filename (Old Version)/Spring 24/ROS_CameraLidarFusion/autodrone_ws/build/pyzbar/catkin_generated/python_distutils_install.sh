#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/zeningli/Documents/autodrone_ws/src/pyzbar"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/zeningli/Documents/autodrone_ws/install/lib/python3.11/site-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/zeningli/Documents/autodrone_ws/install/lib/python3.11/site-packages:/home/zeningli/Documents/autodrone_ws/build/lib/python3.11/site-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/zeningli/Documents/autodrone_ws/build" \
    "/home/zeningli/miniforge3/envs/ros_env/bin/python3.11" \
    "/home/zeningli/Documents/autodrone_ws/src/pyzbar/setup.py" \
     \
    build --build-base "/home/zeningli/Documents/autodrone_ws/build/pyzbar" \
    install \
    --root="${DESTDIR-/}" \
     --prefix="/home/zeningli/Documents/autodrone_ws/install" --install-scripts="/home/zeningli/Documents/autodrone_ws/install/bin"
