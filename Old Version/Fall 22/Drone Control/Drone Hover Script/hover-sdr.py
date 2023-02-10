import time
import argparse
import numpy as np

from rtlsdr import RtlSdr
from skyengine.drone import DroneController
from skylog.logger import DirectLogger
from skylog.message import BaseMessage
from skymission.concurrency import tick
from skymission.mission import Mission
from skymission.mission import callback
from skymission.mission import panic

# represents a message containing the drone's current location, speed, heading direction, and SDR readings
class LocationMessage(BaseMessage):
    """
    Message for the drone's current location, speed, heading direction, and sdr readings.
    """

    def __init__(self, timestamp, location, speed, heading, sdr, dBm):
        self.timestamp = timestamp
        self.lat = location.lat
        self.lon = location.lon
        self.alt = location.alt
        self.speed = speed
        self.heading = heading
        self.sdr = sdr
        self.dBm = dBm

    # to convert the message into a dictionary that can be logged.
    def serialize(self):
        return {
            'timestamp': self.timestamp,
            'location': {
                'lat': self.lat,
                'lon': self.lon,
                'alt': self.alt
            },
            'speed':self.speed,
            'heading':self.heading,
            'IQ': self.sdr,
            'dBm':self.dBm,
        }


# represents a drone flight mission that logs the drone's location, speed, and heading while the drone is flying non-autonomously. The class contains several methods for controlling the drone and logging information.
class HoverMission(Mission):
    """
    This mission will log location, speed, and heading while the drone flys non-autonomously.
    """

    mission_id = 'hover-sdr'
    port = 4002

    # is called when an instance of the HoverMission class is created. This method takes in the address of the flight controller and the name of the log file to use. The method initializes the DroneController and DirectLogger instances and configures the SDR device. It also starts the mission server.
    def __init__(self, fc_addr, log_file):
        """
        Create a TraversalMission and start the mission server.

        :param fc_addr: MAVProxy address of the flight controller.
        :param log_file: Name of the log file for location data.
        """
        self.enable_disk_event_logging()

        self.dc = DroneController(fc_addr)
        self.logger = DirectLogger(path=log_file)
        self.log.debug('Drone controller and logger initialized successfully')

        self.is_mission_complete = False

        self.sdr = RtlSdr()
        # configure device
        self.sdr.sample_rate = 2.048e6  # Hz
        self.sdr.center_freq = 563e6     # Hz
        self.freq_correction = 60   # PPM
        self.sdr.gain = 10
        self.SamplesToDiscard = 10
        self.ContSamples = 0

        self.start_server()


    # it is called periodically by the tick decorator. This method logs the drone's current GPS location, speed, and heading, as well as the SDR readings and signal strength data.
    @tick()
    def start_log(self):
        """
        Start periodically logging the drone GPS location, speed, and direction.
        """
        if self.is_mission_complete:
            return

        nSamples = 256
        samples = self.sdr.read_samples(nSamples)
        self.ContSamples +=1
        if self.ContSamples > self.SamplesToDiscard : 
            dBm = 10 * np.log10(np.mean(np.power(np.abs(samples), 2)))
            gps_pos = self.dc.read_gps()
            sdr = samples.tolist()
            sdr_str = [str(c) for c in sdr]
            self.logger.log(LocationMessage(
                timestamp=time.time(),
                location=gps_pos,
                speed=self.dc.vehicle.airspeed,
                heading=self.dc.vehicle.heading,
                sdr=sdr_str,
                dBm = dBm
            ))

    # called by a client over the network. This method takes in the altitude to hover at and the time to hover for. It tells the drone to take off and hover at the specified altitude for the specified amount of time.
    @callback(
        endpoint='/start-mission',
        description='Start logging the location, speed, and heading of drone while non-autonomous flight occurs.',
        required_params=('alt', 'hover_time'),
        public=True,
    )
    def start_mission(self, data, *args, **kwargs):
        """
        Client invoked endpoint to begin the hover mission. 

        The mission will tell the drone to take off to 5 m and then only continue logging information.

        :param data: Required to be of the form:
                     {
                     }
        """
       
        #begin flight
        current_local = self.dc.read_gps()
        altitudes = data['alt']
        self.log.debug('Taking off: altitude {alt}'.format(alt=altitudes[0]))
        self.dc.take_off(altitudes[0])
        self.start_log()
        self.log.debug('Take off complete and beginning log')
        self.log.debug('Hovering for {} seconds'.format(data['hover_time']))
        time.sleep(data['hover_time'])
        for altitude in altitudes[1:]:
            try:
                self.log.debug('Changing altitude to {}'.format(altitude))
                self.dc.goto(coords=(current_local.lat,current_local.lon), altitude=altitude, airspeed=2)
                self.log.debug('Hovering for {} seconds'.format(data['hover_time']))
                time.sleep(data['hover_time'])
            except FlightAbortedException:
                self.log.error('Flight aborted due to panic; aborting remaining tasks.')
        self.is_mission_complete = True
        self.log.debug('Drone landing')
        self.dc.land()
        

    @panic
    def panic(self, *args, **kwargs):
        self.log.info('Mission panicked! Landing immediately.')
        self.dc.panic()


# This function parses command-line arguments and creates an instance of the HoverMission class. It then waits for the mission to complete before exiting.
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
        default='{year}_{month}_{day}_{hour}_{min}_{sec}-travel-path.log'.format(
            year=time.localtime()[0],
            month=time.localtime()[1],
            day=time.localtime()[2],
            hour=time.localtime()[3],
            min=time.localtime()[4],
            sec=time.localtime()[5],
            ),  
    )
    args = parser.parse_args()

    HoverMission(
        fc_addr=args.fc_addr,
        log_file=args.log_file,
    )
