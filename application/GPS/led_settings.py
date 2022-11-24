import enum
import threading
import time
import os
# from Jetson import GPIO
from application.utils.settings import GPS_conf


class LedColor(enum.Enum):
    GREEN = 0
    RED = 1
    ORANGE = 2


class LedState(enum.Enum):
    OFF = 0
    ON = 1
    BLINK = 2


class LedSettings:
    _state = LedState.OFF
    _color = LedColor.ORANGE

    @staticmethod
    def turn_on(color: LedColor):
        if color == LedSettings._color and LedSettings._state != LedState.OFF: return
        LedSettings.turn_off()
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.ON
        LedSettings._color = color
        # os.system(f'/home/mic-710aix/fruitspec/FS-ZED/GPS/diotest -set {color.value} {GPIO.HIGH}')

    @staticmethod
    def turn_off():
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.OFF
        # os.system(f'/home/mic-710aix/fruitspec/FS-ZED/GPS/diotest -set {LedColor.GREEN.value} {GPIO.LOW}')
        # os.system(f'/home/mic-710aix/fruitspec/FS-ZED/GPS/diotest -set {LedColor.RED.value} {GPIO.LOW}')
        # os.system(f'/home/mic-710aix/fruitspec/FS-ZED/GPS/diotest -set {LedColor.ORANGE.value} {GPIO.LOW}')

    @staticmethod
    def start_blinking(color_led, repeat_time=2, pause_time=0.5):
        LedSettings._state = LedState.BLINK
        t1 = threading.Thread(target=LedSettings._set_blinking,
                              args=(color_led, repeat_time, pause_time), daemon=True)
        t1.start()

    @staticmethod
    def stop_blinking():
        LedSettings._state = LedState.OFF

    @staticmethod
    def _set_blinking(color: LedColor, repeat_time, pause_time):
        LedSettings._color = color
        while LedSettings._state == LedState.BLINK:
            for i in range(repeat_time):
                LedSettings.turn_on(color)
                time.sleep(GPS_conf["led blink sleep time"])
                LedSettings.turn_off()
                time.sleep(GPS_conf["led blink sleep time"])

            LedSettings.turn_on(color)
            time.sleep(pause_time)
            LedSettings.turn_off()
