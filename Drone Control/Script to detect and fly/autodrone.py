import argparse
import time
import math


from skyengine.drone import DroneController
from skyengine.exceptions import FlightAbortedException
from skylog.logger import DirectLogger
from skylog.message import BaseMessage
from skymission.concurrency import tick
from skymission.mission import Mission
from skymission.mission import callback
from skymission.mission import panic


class Waypoint:
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt


class LocationMessage(BaseMessage):
    """
    Message for the drone's current location.
    """

    def __init__(self, timestamp, lat, lon, alt):
        self.timestamp = timestamp
        self.lat = lat
        self.lon = lon
        self.alt = alt

    def serialize(self):
        return {
            'timestamp': self.timestamp,
            'lat': self.lat,
            'lon': self.lon,
            'alt': self.alt,
        }

    @staticmethod
    def deserialize(json):
        return LocationMessage(
            timestamp=json['timestamp'],
            lat=json['lat'],
            lon=json['lon'],
            alt=json['alt'],
        )


class WaypointGotoMission(Mission):

    port = 4002
    mission_id = 'waypoint-goto'

    def __init__(self, fc_addr, log_file):
        """
        Create a WaypointGotoMission and start the mission server.

        :param fc_addr: MAVProxy address of the flight controller.
        :param log_file: Name of the log file for location data.
        """
        self.enable_disk_event_logging()

        self.dc = DroneController(fc_addr)
        self.location_logger = DirectLogger(path=log_file)
        self.log.debug('Drone controller and logger initialized successfully')

        self.cancel_tick = self.start_location_log()
        self.log.info('Waypoint-goto mission initialization complete')

        self.start_server()

    @tick(interval=0.5)
    def start_location_log(self):
        """
        Start periodically logging the drone GPS location to disk.
        """
        location = self.dc.read_gps()
        message = LocationMessage(
            timestamp=time.time(),
            lat=location.lat,
            lon=location.lon,
            alt=location.alt,
        )

        self.location_logger.log(message)

    @callback(
        endpoint='/start-mission',
        description='Gives the drone a series of waypoints and starts the mission.',
        required_params=('waypoints', 'hover_duration'),
        public=True,
    )
    def start_mission(self, data, *args, **kwargs):
        """

        :param data: Required to be of the form:
                     [{
                         'lat': ...,  # Target latitude
                         'lon': ...,  # Target longitude
                         'alt': ...,  # Target altitude
                     }]
        """
        try:
            hover_duration = data['hover_duration']
            waypoints = [
                Waypoint(point['lat'], point['lon'], point['alt'])
                for point in data['waypoints']
            ]
            start_alt = waypoints[0].alt

            self.log.debug('Taking off to altitude: {alt}'.format(alt=start_alt))
            self.dc.take_off(start_alt)
            self.log.debug('Take off complete')

            for i in range(6):
                self.dc.move_forward(0.5)
                time.sleep(5)

            for waypoint in waypoints:
                self.log.debug('Navigating to waypoint: ({lat}, {lat})'.format(
                    lat=waypoint.lat,
                    lon=waypoint.lon,
                ))
                self.dc.goto(coords=(waypoint.lat, waypoint.lon), altitude=waypoint.alt, airspeed=5)
                self.log.debug('Navigation to waypoint complete')

                location = self.dc.read_gps()
                self.log.debug('Arrived! Current location: ({lat}, {lon})'.format(
                    lat=location.lat,
                    lon=location.lon,
                ))

                self.log.debug('Hovering for {hover_duration} seconds'.format(
                    hover_duration=hover_duration,
                ))
                time.sleep(hover_duration)

            self.log.info('Navigation to all waypoints complete. Landing now.')
            self.dc.land()
            self.log.info('Landed!')
        except FlightAbortedException:
            self.log.error('Flight aborted due to panic; aborting remaining tasks.')

    @panic
    def panic(self, *args, **kwargs):
        self.log.info('Mission panicked! Landing immediately.')
        self.dc.panic()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fc-addr',
        dest='fc_addr',
        help='Address of the flight controller mavproxy messages',
        default=None,
    )
    parser.add_argument(
        '--log-file',
        dest='log_file',
        help='Path to the log file to create',
        default='travel-path',
    )
    args = parser.parse_args()

    WaypointGotoMission(
        fc_addr=args.fc_addr,
        log_file=args.log_file,
    )




#     mission.start_mission(
#     waypoints=[        {'lat': 37.4220, 'lon': -122.0841, 'alt': 0},        {'lat': 37.4221, 'lon': -122.0842, 'alt': 0},        {'lat': 37.4222, 'lon': -122.0843, 'alt': 0},    ],
#     hover_duration=2,
# )





# import math

# def calculate_new_gps_location(current_gps_location, distance, num_times):
#     # Extract the current latitude and longitude from the current GPS location
#     lat, lng = current_gps_location

#     # Calculate the new latitude by adding the distance to the current latitude,
#     # multiplied by the number of times the function is called
#     new_lat = lat + (distance * num_times)

#     # Calculate the new longitude by adding the distance * the cosine of the current latitude
#     # to the current longitude, multiplied by the number of times the function is called
#     new_lng = lng + (distance * math.cos(lat) * num_times)

#     # Return the new GPS location as a tuple of latitude and longitude
#     return (new_lat, new_lng)


# # Set the current GPS location of the drone
# current_gps_location = (37.4220, -122.0841)

# # Calculate the new GPS location by moving north for 0.5 meters 4 times
# new_gps_location = calculate_new_gps_location(current_gps_location, 0.5, 4)

# # Print the new GPS location to the console
# print(new_gps_location)





# # Import the time module for sleeping
# import time

# # Set the initial position of the drone
# initial_position = (37.4220, -122.0841)

# # Create a DroneController object
# dc = DroneController(...)

# # Set the distance flown to 0
# distance_flown = 0

# # Fly north in increments of 0.5 meters until 6 meters have been flown
# while distance_flown < 6:
#     # Calculate the new position by adding 0.5 to the initial latitude
#     new_position = (initial_position[0] + 0.5, initial_position[1])

#     # Fly to the new position at an airspeed of 5 m/s
#     dc.goto(new_position, 0, 5)

#     # Add 0.5 to the total distance flown
#     distance_flown += 0.5

#     # Sleep for 5 seconds
#     time.sleep(5)

# # The drone has now flown 6 meters to the north



# def start_mission(self, data, *args, **kwargs):
#     """
#     Move the drone north for 0.5 meters every 5 seconds.
#     """
#     total_distance = 0
#     while total_distance < 6:
#         self.dc.relative_move(direction='north', distance=0.5)
#         total_distance += 0.5
#         time.sleep(5)

#     self.log.info('Navigation complete. Landing...')
#     self.dc.land()





# @callback(
#     endpoint='/start-mission',
#     description='Gives the drone a series of waypoints and starts the mission.',
#     required_params=('waypoints', 'hover_duration'),
#     public=True,
# )
# def start_mission(self, data, *args, **kwargs):
#     try:
#         hover_duration = data['hover_duration']

#         self.log.debug('Taking off')
#         self.dc.take_off()
#         self.log.debug('Take off complete')

#         for i in range(6):
#             self.dc.move_forward(0.5)
#             time.sleep(5)

#         self.log.info('Moving forward 6 meters complete. Landing...
#     except Exception as e:
#         self.log.error('Error in start_mission: {e}'.format(e=str(e)))
#         return self.status_error('Error starting mission')

# This modified start_mission function takes off, moves forward 0.5 meters every 5 seconds for a total 
# of 6 * 0.5 = 3 meters, and then lands. It no longer uses the waypoints and hover_duration parameters from 
# the data input, since these are not needed for the updated mission.




# Start the script: python waypoint_goto.py --fc-addr=127.0.0.1 --log-file=location.log
#  def generate_waypoints(altitude, distance):
#     waypoints = []

#     for i in range(3):
#         waypoints.append({
#             "lat": 0,
#             "lon": 0,
#             "alt": altitude,
#         })
#         waypoints.append({
#             "lat": distance,
#             "lon": 0,
#             "alt": altitude,
#         })

#     return waypoints



# waypoints = generate_waypoints(10, 0.5)


# curl -X POST http://localhost:4002/start-mission -d '{
#   "waypoints": waypoints,
#   "hover_duration": 5
# }'


