import logging
import subprocess
import threading
from enum import Enum

from application.utils.module_wrapper import ModulesEnum, Module
from application.utils.settings import GUI_conf, conf, analysis_conf
from eventlet import listen as wsgi_listen
from eventlet.wsgi import server as wsgi_server
import socketio
import os
import usb.core
import netifaces
import json


class DeviceStates:
    ON = "On"
    OFF = "Off"


class GUIInterface(Module):
    gps_state = DeviceStates.OFF
    jai_state = DeviceStates.OFF
    zed_state = DeviceStates.OFF
    listener = None
    server_thread = None
    sio = socketio.Server(cors_allowed_origins='*')

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        def setup_server():
            app = socketio.WSGIApp(GUIInterface.sio)
            wsgi_server(GUIInterface.listener, app)

        if not conf["GUI"]:
            return False

        super(GUIInterface, GUIInterface).init_module(sender, receiver, main_pid, module_name)
        super(GUIInterface, GUIInterface).set_signals(GUIInterface.shutdown, GUIInterface.receive_data)

        GUIInterface.listener = wsgi_listen(('', GUI_conf["GUI server port"]))
        subprocess.Popen(GUI_conf["GUI startup script"], shell=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        GUIInterface.server_thread = threading.Thread(target=setup_server, daemon=True)
        GUIInterface.server_thread.start()
        GUIInterface.server_thread.join()

    # default events

    @staticmethod
    @sio.event
    def receive_data(sid, environ):
        data, sender_module = GUIInterface.receiver.recv()
        action, data = data["action"], data["data"]
        if sender_module == ModulesEnum.Analysis:
            jai_connected, zed_connected = data
            GUIInterface.jai_state = DeviceStates.ON if jai_connected else DeviceStates.OFF
            GUIInterface.zed_state = DeviceStates.ON if zed_connected else DeviceStates.OFF
            logging.log(logging.INFO, f"SET CAMERAS STATE: JAI -> {GUIInterface.jai_state},"
                                      f" ZED -> {GUIInterface.zed_state}")

    @staticmethod
    @sio.event
    def connect(sid, environ):
        logging.info(f"CONNECTION ESTABLISHED: {sid}")
        states = json.dumps({
            'JAI': GUIInterface.jai_state,
            "ZED": GUIInterface.zed_state
        })
        GUIInterface.sio.emit('set_camera_state', states)

    @staticmethod
    @sio.event
    def disconnect_gui(sid):
        logging.info(f"DISCONNECTED GUI: {sid}")

    # custom events

    @staticmethod
    @sio.event
    def start_recording(sid, data):
        logging.info(f"CAMERA START RECORDING RECEIVED: {sid}, data {data}")
        GUIInterface.send_data("start", data, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def start_view(sid):
        logging.log(logging.INFO, f"CAMERA START VIEW RECEIVED: {sid}")
        GUIInterface.send_data("view", None, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def stop_recording(sid, data):
        logging.log(logging.INFO, f"CAMERA STOP RECEIVED: {sid}")
        GUIInterface.send_data("stop", None, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def stop_view(sid, data):
        logging.log(logging.INFO, f"CAMERA STOP RECEIVED: {sid}")
        GUIInterface.send_data("stop", None, ModulesEnum.Analysis)

    @staticmethod
    def set_GPS_state(state):
        logging.log(logging.INFO, f"SET GPS STATE TO {state}")
        GUIInterface.gps_state = state
        GUIInterface.sio.emit('set_GPS_state', json.dumps({'gps': state}))

    @staticmethod
    def set_camera_custom_config(FPS, integration_time, output_fsi, output_rgb, output_800, output_975, output_svo):
        logging.log(logging.INFO, f"SET CAMERA CONFIG")
        camera_config = {
            'FPS': FPS,
            'integration time': integration_time,
            'output_fsi': output_fsi,
            'output_rgb': output_rgb,
            'output_800': output_800,
            'output_975': output_975,
            'output_svo': output_svo
        }
        # trigger gui
        GUIInterface.sio.emit('get_camera_custom_config', json.dumps(camera_config))

    @staticmethod
    def set_camera_easy_config(weather):
        logging.log(logging.INFO, f"SET CAMERA CONFIG")
        # trigger gui
        GUIInterface.sio.emit('get_camera_state', json.dumps({'weather': weather}))
