import argparse
import time

from skyengine.drone import DroneController
from skyengine.exceptions import FlightAbortedException
from skylog.logger import DirectLogger
from skylog.message import BaseMessage
from skymission.concurrency import tick
from skymission.mission import Mission
from skymission.mission import callback
from skymission.mission import panic


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


class HoverMission(Mission):
    """
    A mission to take off, hover, and land.
    """

    port = 4000
    mission_id = 'hover'

    # In the __init__ method, a DroneController instance is initialized with the given flight controller address and a DirectLogger instance is initialized to log location data to a specified file.
    def __init__(self, fc_addr, log_file):
        """
        Create a HoverMission, which is started as soon as the class is instantiated.

        :param fc_addr: MAVProxy address of the flight controller.
        :param log_file: Name of log file for location data.
        """
        self.enable_disk_event_logging()

        self.dc = DroneController(fc_addr)
        self.location_logger = DirectLogger(path=log_file)
        self.log.debug('Drone controller and logger initialized successfully')

        self.cancel_tick = self.start_location_log()
        self.log.info('Hover mission initialization complete')

        self.start_server()

    # set to run periodically to log the drone's current location.
    @tick(interval=0.5)
    def start_location_log(self):
        """
        Start periodically logging the drone GPS location to disk.
        """
        location = self.dc.vehicle.location.global_frame
        message = LocationMessage(
            timestamp=time.time(),
            lat=location.lat,
            lon=location.lon,
            alt=location.alt,
        )

        # self.log.debug('Recording current location: ({lat}, {lon})'.format(
        #     lat=location.lat,
        #     lon=location.lon,
        # ))

        self.location_logger.log(message)

    # a callback endpoint that is called by a client to start the hover mission. It takes off, hovers for the specified amount of time at the specified altitude, and lands.
    @callback(
        endpoint='/start-mission',
        description='Gives the drone an altitude and hover time.',
        required_params=('alt', 'hover_time'),
        public=True,
    )
    def start_mission(self, data, *args, **kwargs):
        """
        Client-invoked endpoint to begin the hover mission.

        :param data: Required to be of the form:
                     {
                         'alt': ...,  # Target altitude (m)
                         'hover_time': ..., # Hover time (s)
                     }
        """
        alt = data['alt']
        hover_time = data['hover_time']

        try:
            self.log.debug('Taking off to altitude: {alt}'.format(alt=alt))
            self.dc.take_off(alt)
            self.log.debug('Take off complete')

            self.log.debug('Hovering for: {hover_time} seconds'.format(hover_time=hover_time))
            time.sleep(hover_time)

            self.log.info('Hovering complete; begin landing')
            self.dc.land()
            self.log.info('Landed!')
        except FlightAbortedException:
            self.log.warn('Flight aborted due to emergency panic!')

    # decorated with the @panic decorator, which allows it to be called as a panic endpoint to immediately abort the mission and land the drone.
    @panic
    def panic(self, *args, **kwargs):
        self.log.warn('Panic request received')
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

    HoverMission(
        fc_addr=args.fc_addr,
        log_file=args.log_file,
    )
