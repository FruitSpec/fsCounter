import enum
import threading
import time
import os
from Jetson import GPIO
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
    def turn_on(color: LedColor, stop_blinking=True):
        # print(f"turn on color: {LedSettings._color}")
        # print(f"turn on state: {LedSettings._state}")
        if color == LedSettings._color and LedSettings._state == LedState.ON:
            return
        if stop_blinking:
            LedSettings._state = LedState.OFF
        os.system(f"{GPS_conf.led_exe_path} -set {color.value} {GPIO.HIGH}")
        LedSettings._color = color
        LedSettings.turn_off(exclude=color)
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.ON
        LedSettings._color = color

    @staticmethod
    def turn_off(exclude=None):
        # print(f"turn off color: {LedSettings._color}")
        # print(f"turn off state: {LedSettings._state}")
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.OFF
        if exclude != LedColor.GREEN:
            os.system(f"{GPS_conf.led_exe_path} -set {LedColor.GREEN.value} {GPIO.LOW}")
        if exclude != LedColor.ORANGE:
            os.system(f"{GPS_conf.led_exe_path} -set {LedColor.ORANGE.value} {GPIO.LOW}")
        if exclude != LedColor.RED:
            os.system(f"{GPS_conf.led_exe_path} -set {LedColor.RED.value} {GPIO.LOW}")

    @staticmethod
    def start_blinking(*colors):
        #print(f"blinking color: {LedSettings._color}")
        #print(f"blinking state: {LedSettings._state}")
        if LedSettings._state == LedState.BLINK:
            return
        LedSettings._state = LedState.BLINK

        def set_blinking():
            i = 0
            while LedSettings._state == LedState.BLINK:
                LedSettings._color = colors[i % len(colors)]
                i += 1
                LedSettings.turn_on(LedSettings._color, stop_blinking=False)
                time.sleep(GPS_conf.led_blink_sleep_time)

        t1 = threading.Thread(target=set_blinking, daemon=True)
        t1.start()
