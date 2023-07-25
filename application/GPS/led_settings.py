import enum
import threading
import time
import logging
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
    def turn_on(color: LedColor):
        if color == LedSettings._color and LedSettings._state != LedState.OFF:
            return
        logging.info(f"LED TURN ON {color.name}")
        print(f"LED TURN ON {color.name}")
        LedSettings.turn_off()
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.ON
        LedSettings._color = color
        os.system(f"{GPS_conf.led_exe_path} -set {color.value} {GPIO.HIGH}")

    @staticmethod
    def turn_off():
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.OFF
        logging.info("LED TURN OFF")
        print("LED TURN OFF")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.GREEN.value} {GPIO.LOW}")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.RED.value} {GPIO.LOW}")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.ORANGE.value} {GPIO.LOW}")

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
