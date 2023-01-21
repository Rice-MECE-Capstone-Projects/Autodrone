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
