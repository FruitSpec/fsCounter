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
    BLINK_TRANSPARENT = 3


class LedState(enum.Enum):
    OFF = 0
    ON = 1
    BLINK = 2


class LedSettings:
    _state = LedState.OFF
    _color = LedColor.ORANGE

    @staticmethod
    def turn_on(color: LedColor, should_print=True):
        if color == LedSettings._color and LedSettings._state != LedState.OFF:
            return
        if should_print:
            logging.info(f"LED TURN ON {color.name}")
            print(f"LED TURN ON {color.name}")
        LedSettings.turn_off(should_print=False)
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.ON
        LedSettings._color = color
        if LedSettings._color != LedColor.BLINK_TRANSPARENT:
            os.system(f"{GPS_conf.led_exe_path} -set {color.value} {GPIO.HIGH}")

    @staticmethod
    def turn_off(should_print=True):
        if LedSettings._state != LedState.BLINK:
            LedSettings._state = LedState.OFF
        if should_print:
            logging.info("LED TURN OFF")
            print("LED TURN OFF")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.GREEN.value} {GPIO.LOW}")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.RED.value} {GPIO.LOW}")
        os.system(f"{GPS_conf.led_exe_path} -set {LedColor.ORANGE.value} {GPIO.LOW}")

    @staticmethod
    def start_blinking(*colors):
        if LedSettings._state == LedState.BLINK:
            return
        LedSettings._state = LedState.BLINK

        def set_blinking():
            logging.info(f"LED START BLINKING {list(colors)}")
            print(f"LED START BLINKING {list(colors)}")

            i = 0
            while LedSettings._state == LedState.BLINK:
                LedSettings._color = colors[i % len(colors)]
                i += 1
                LedSettings.turn_on(LedSettings._color, should_print=False)
                time.sleep(GPS_conf.led_blink_sleep_time)

            logging.info(f"LED STOP BLINKING")
            print(f"LED STOP BLINKING")

        t1 = threading.Thread(target=set_blinking, daemon=True)
        t1.start()
