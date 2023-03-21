import logging
import subprocess
import threading
from enum import Enum

from application.utils.module_wrapper import ModulesEnum, Module
from application.utils.settings import GUI_conf, conf
from eventlet import listen as wsgi_listen
from eventlet.wsgi import server as wsgi_server
import socketio
import os
import usb.core
import netifaces
import json


class DeviceStates(Enum):
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
    def init_module(sender, receiver, main_pid):
        def setup_server():
            app = socketio.WSGIApp(GUIInterface.sio)
            wsgi_server(GUIInterface.listener, app)

        if not conf["GUI"]:
            return False

        super(GUIInterface, GUIInterface).init_module(sender, receiver, main_pid)
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
    def set_connections():
        GUIInterface.set_GPS_state(DeviceStates.OFF)
        try:
            jai_ip = netifaces.ifaddresses('eth2')[netifaces.AF_INET][0]['addr']
            jai_state = DeviceStates.ON
        except Exception:
            jai_state = DeviceStates.OFF

        zed_state = DeviceStates.OFF
        usbdev = usb.core.find(find_all=True)
        for dev in usbdev:
            try:
                manufacturer = usb.util.get_string(dev, dev.iManufacturer)
                product = usb.util.get_string(dev, dev.iProduct)
                if 'STEREOLABS' in manufacturer and 'ZED' in product:
                    dev.get_active_configuration()
                    zed_state = DeviceStates.ON
                    break
            except Exception:
                pass
        GUIInterface.set_cameras_state(jai_state, zed_state)

    @staticmethod
    @sio.event
    def connect(sid, environ):
        logging.info(f"GUI CONNECTION ESTABLISHED: {sid}")
        GUIInterface.set_connections()

    @staticmethod
    @sio.event
    def disconnect(sid):
        logging.info(f"DISCONNECTED GUI: {sid}")

    # custom events

    @staticmethod
    @sio.event
    def start_recording(sid, data):
        if 'Default' in data['configType']:
            weather = data['weather']
            camera_data = conf["default camera data"][weather]
            output_types = conf["default output types"]
        else:
            camera_data = data["Cameras"]
            output_types = camera_data["outputTypes"]

        output_types = [ot.lower() for ot in output_types]

        fps = int(camera_data['FPS'])
        exposure_rgb = int(camera_data['IntegrationTimeRGB'])
        exposure_800 = int(camera_data['IntegrationTime800'])
        exposure_975 = int(camera_data['IntegrationTime975'])
        output_dir = os.path.join('/' + data['outputPath'], data['plot'], 'row_' + data['row'])
        output_fsi = 'fsi' in output_types
        output_rgb = 'rgb' in output_types
        output_800 = '800' in output_types
        output_975 = '975' in output_types
        output_svo = 'svo' in output_types
        view = False
        use_clahe_stretch = False
        debug_mode = True

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.info(f"GUI CAMERA START RECORDING RECEIVED: {sid}, data {data}")
        data_dict = {
            "action": "start",
            "data": data
        }
        GUIInterface.send_data(data_dict, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def start_view(sid):
        logging.log(logging.INFO, f"GUI CAMERA START VIEW RECEIVED: {sid}")
        data_dict = {"action": "view"}
        GUIInterface.send_data(data_dict, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def stop_recording(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA STOP RECEIVED: {sid}")
        data_dict = {"action": "stop"}
        GUIInterface.send_data(data_dict, ModulesEnum.Analysis)

    @staticmethod
    @sio.event
    def stop_view(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA STOP RECEIVED: {sid}")
        data_dict = {"action": "stop"}
        GUIInterface.send_data(data_dict, ModulesEnum.Analysis)

    @staticmethod
    def set_GPS_state(state):
        logging.log(logging.INFO, f"GUI SET GPS STATE TO {state}")
        GUIInterface.gps_state = state
        GUIInterface.sio.emit('set_GPS_state', json.dumps({'gps': state.value}))

    @staticmethod
    def set_cameras_state(jai_state, zed_state):
        logging.log(logging.INFO, f"GUI SET CAMERAS STATE: JAI -> {jai_state} | ZED -> {zed_state}")
        GUIInterface.jai_state = jai_state
        GUIInterface.zed_state = zed_state
        GUIInterface.sio.emit('set_camera_state', json.dumps({'JAI': jai_state.value, "ZED": zed_state.value}))

    @staticmethod
    def set_camera_custom_config(FPS, integration_time, output_fsi, output_rgb, output_800, output_975, output_svo):
        logging.log(logging.INFO, f"GUI SET CAMERA CONFIG")
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
        logging.log(logging.INFO, f"GUI SET CAMERA CONFIG")
        # trigger gui
        GUIInterface.sio.emit('get_camera_state', json.dumps({'weather': weather}))
