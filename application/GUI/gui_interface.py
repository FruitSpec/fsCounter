import logging
import subprocess
import threading

from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils.settings import GUI_conf, conf
from application.utils.settings import set_logger
from application.utils import tools
from eventlet import listen as wsgi_listen
from eventlet.wsgi import server as wsgi_server
import socketio
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
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        def setup_server():
            app = socketio.WSGIApp(GUIInterface.sio)
            wsgi_server(GUIInterface.listener, app)

        if not conf.GUI:
            return False

        super(GUIInterface, GUIInterface).init_module(in_qu, out_qu, main_pid, module_name,
                                                      communication_queue, notify_on_death, death_action)
        super(GUIInterface, GUIInterface).set_signals(GUIInterface)

        GUIInterface.listener = wsgi_listen(('', GUI_conf.GUI_server_port))
        subprocess.Popen(GUI_conf.GUI_startup_script, shell=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)

        GUIInterface.receive_data_thread = threading.Thread(target=GUIInterface.receive_data, daemon=True)
        GUIInterface.server_thread = threading.Thread(target=setup_server, daemon=True)

        GUIInterface.receive_data_thread.start()
        GUIInterface.server_thread.start()

        GUIInterface.receive_data_thread.join()
        GUIInterface.server_thread.join()

    # default events

    @staticmethod
    @sio.event
    def receive_data(sid=None, environ=None):
        while not GUIInterface.shutdown_event.is_set():
            data, sender_module = GUIInterface.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.Acquisition:
                if action == ModuleTransferAction.GUI_SET_DEVICE_STATE:
                    jai_connected, zed_connected = data
                    GUIInterface.jai_state = DeviceStates.ON if jai_connected else DeviceStates.OFF
                    GUIInterface.zed_state = DeviceStates.ON if zed_connected else DeviceStates.OFF
                    states = json.dumps({
                        "JAI": GUIInterface.jai_state,
                        "ZED": GUIInterface.zed_state
                    })
                    GUIInterface.sio.emit('set_camera_state', states)
                    tools.log(
                        f"SET CAMERAS STATE: JAI -> {GUIInterface.jai_state},"
                        f" ZED -> {GUIInterface.zed_state}"
                    )
            if sender_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    GUIInterface.send_data(
                        action=ModuleTransferAction.MONITOR,
                        data=None,
                        receiver=ModulesEnum.Main,
                        log_option=tools.LogOptions.LOG
                    )
                elif action == ModuleTransferAction.SET_LOGGER:
                    set_logger()

    @staticmethod
    @sio.event
    def connect_gui(sid, environ):
        tools.log(f"CONNECTION ESTABLISHED: {sid}")
        states = json.dumps({
            'JAI': GUIInterface.jai_state,
            "ZED": GUIInterface.zed_state
        })
        GUIInterface.sio.emit('set_camera_state', states)

    @staticmethod
    @sio.event
    def disconnect_gui(sid, environ):
        tools.log(f"DISCONNECTED GUI: {sid}")

    # custom events

    @staticmethod
    @sio.event
    def start_recording(sid, data):
        tools.log(f"CAMERA START RECORDING RECEIVED: {sid}, data {data}")
        GUIInterface.send_data(ModuleTransferAction.START_ACQUISITION, data, ModulesEnum.Acquisition)

    @staticmethod
    @sio.event
    def start_view(sid):
        tools.log(f"CAMERA START VIEW RECEIVED: {sid}")
        GUIInterface.send_data(ModuleTransferAction.VIEW_START, None, ModulesEnum.Acquisition)

    @staticmethod
    @sio.event
    def stop_recording(sid, data):
        tools.log(f"CAMERA STOP RECEIVED: {sid}")
        GUIInterface.send_data(ModuleTransferAction.STOP_ACQUISITION, None, ModulesEnum.Acquisition)

    @staticmethod
    @sio.event
    def stop_view(sid, data):
        tools.log(f"CAMERA STOP RECEIVED: {sid}")
        GUIInterface.send_data(ModuleTransferAction.VIEW_STOP, None, ModulesEnum.Acquisition)

    @staticmethod
    def set_GPS_state(state):
        tools.log(f"SET GPS STATE TO {state}")
        GUIInterface.gps_state = state
        GUIInterface.sio.emit('set_GPS_state', json.dumps({'gps': state}))

    @staticmethod
    def set_camera_custom_config(FPS, integration_time, output_fsi, output_rgb, output_800, output_975, output_svo):
        tools.log(f"SET CAMERA CONFIG")
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
        tools.log(f"SET CAMERA CONFIG")
        # trigger gui
        GUIInterface.sio.emit('get_camera_state', json.dumps({'weather': weather}))
