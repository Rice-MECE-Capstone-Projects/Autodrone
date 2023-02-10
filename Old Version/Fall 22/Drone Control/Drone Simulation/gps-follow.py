import argparse
import time

from geo import Coordinate
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


class GpsFollowMission(Mission):

    port = 4010
    mission_id = 'gps-follow'

    def __init__(self, fc_addr, log_file):
        """
        Create a GpsFollowMission and start the mission server.

        :param fc_addr: MAVProxy address of the flight controller.
        :param log_file: Name of the log file for location data.
        """
        self.AIR_SPEED = 2.0
        self.STEP_SIZE = 3
        self.END_MISSION_DELAY = 5

        self.enable_disk_event_logging()

        self.dc = DroneController(fc_addr)
        self.location_logger = DirectLogger(path=log_file)
        self.log.debug('Drone controller and logger initialized successfully')

        self.cancel_tick = self.start_location_log()
        self.log.info('Gps-follow mission initialization complete')

        self.target_location = Coordinate.from_gps_data(self.dc.read_gps())
        self.end_mission = False

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
        required_params=('alt',),
        public=True
    )
    def start_mission(self, data, *args, **kwargs):
        """
        Start GPS follow mission. The drone will take off and hover at the specified altitude. The
        entire mission will take place at this altitude.

        :param data: Required to be of the form:
                     {'alt': ... # Mission altitude}
        """
        self.altitude = data['alt']

        try:
            self.log.debug('Taking off to altitude: {alt}'.format(alt=self.altitude))
            self.dc.take_off(self.altitude)
            self.log.debug('Take off complete')

            while not self.end_mission:
                # Step towards the target.
                step_coord = self._compute_step_location(self.target_location)
                self.dc.goto(
                    coords=step_coord.pair(),
                    altitude=self.altitude,
                    airspeed=self.AIR_SPEED
                )

            self.log.info('Landing in {t}s.'.format(t=self.END_MISSION_DELAY))
            time.sleep(self.END_MISSION_DELAY)
            self.dc.land()
            self.log.info('Landed!')

        except FlightAbortedException:
            self.log.error('Flight aborted due to panic.')

    def _compute_step_location(self, target):
        """
        Given some target coordinate, travel at most STEP_SIZE meters towards that location.
        """
        here = Coordinate.from_gps_data(self.dc.read_gps())
        d = here.distance_to(target)
        self.log.debug('Distance to target {dist}m.'.format(dist=d))
        if d > self.STEP_SIZE:
            return here.offset_toward_target(target, self.STEP_SIZE)
        return target

    @callback(
        endpoint='/head-to',
        required_params=('lat', 'lon'),
    )
    def head_to(self, data, *args, **kwargs):
        """
        Update the target gps location.

        :param data: Required to be of the form:
                     {
                        'lat': ... # Target latitude
                        'lon': ... # Target longitude
                     }
        """
        self.log.info('Heading to lat: {lat}, lon: {lon}'.format(lat=data['lat'], lon=data['lon']))
        self.target_location = Coordinate(data['lat'], data['lon'])

    @callback(
        endpoint='/end-mission',
        public=True
    )
    def end_mission(self, data, *args, **kwargs):
        """
        End the mission gracefully. The drone will hover for END_MISSION_DELAY seconds before
        landing so that the GPS signal has time to move away.
        """
        self.log.info('Gracefully ending mission.')
        self.end_mission = True

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

    GpsFollowMission(
        fc_addr=args.fc_addr,
        log_file=args.log_file,
    )
