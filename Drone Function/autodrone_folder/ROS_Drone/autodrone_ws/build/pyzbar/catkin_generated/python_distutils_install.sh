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

echo_and_run cd "/home/autodrone/autodrone_folder/autodrone_ws/src/pyzbar"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/autodrone/autodrone_folder/autodrone_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/autodrone/autodrone_folder/autodrone_ws/install/lib/python3/dist-packages:/home/autodrone/autodrone_folder/autodrone_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/autodrone/autodrone_folder/autodrone_ws/build" \
    "/usr/bin/python3" \
    "/home/autodrone/autodrone_folder/autodrone_ws/src/pyzbar/setup.py" \
     \
    build --build-base "/home/autodrone/autodrone_folder/autodrone_ws/build/pyzbar" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/autodrone/autodrone_folder/autodrone_ws/install" --install-scripts="/home/autodrone/autodrone_folder/autodrone_ws/install/bin"
