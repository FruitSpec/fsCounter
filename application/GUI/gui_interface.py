import logging
import subprocess
from application.utils.settings import GUI_conf, conf
from eventlet import listen as wsgi_listen
from eventlet.wsgi import server as wsgi_server
import socketio

import json

sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)


class GUIInterface:
    _connect, _disconnect, _camera_start, _camera_stop = None, None, None, None
    _recorder = -1

    @staticmethod
    def start_GUI(connect, disconnect, camera_start, camera_stop):
        if not conf["GUI"]:
            return False

        listener = wsgi_listen(('', GUI_conf["GUI server port"]))
        GUIInterface._connect = connect
        GUIInterface._disconnect = disconnect
        GUIInterface._camera_start = camera_start
        GUIInterface._camera_stop = camera_stop
        subprocess.Popen(GUI_conf["GUI startup script"], shell=True,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        wsgi_server(listener, app)

    # default events

    @staticmethod
    @sio.event
    def connect(sid, environ):
        logging.log(logging.INFO, f"GUI CONNECTION ESTABLISHED: {sid}")
        jai_on, zed_on, gps_on = GUIInterface._connect(sid, environ)
        jai_state = "On" if jai_on else "Off"
        zed_state = "On" if zed_on else "Off"
        gps_state = "On" if gps_on else "Off"
        sio.emit('set_camera_state', json.dumps({'JAI': jai_state, "ZED": zed_state}))
        sio.emit('set_GPS_state', json.dumps({'gps': gps_state}))

    @staticmethod
    @sio.event
    def disconnect(sid):
        logging.log(logging.INFO, f"DISCONNECTED GUI: {sid}")
        GUIInterface._disconnect()

    # custom events

    @staticmethod
    @sio.event
    def start_recording(sid, data):
        logging.info(f"GUI CAMERA START RECORDING RECEIVED: {sid}, data {data}")
        GUIInterface._recorder = GUIInterface._camera_start(mode="record", data=data)

    @staticmethod
    @sio.event
    def start_view(sid):
        logging.log(logging.INFO, f"GUI CAMERA START VIEW RECEIVED: {sid}")
        GUIInterface._camera_start(mode="view")

    @staticmethod
    @sio.event
    def stop_recording(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA STOP RECEIVED: {sid}")
        GUIInterface._camera_stop(GUIInterface._recorder)

    @staticmethod
    @sio.event
    def stop_view(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA STOP RECEIVED: {sid}")
        GUIInterface._camera_stop(sid, data)

    @staticmethod
    def set_GPS_state(state):
        logging.log(logging.INFO, f"GUI SET GPS STATE TO {state}")
        # trigger gui
        sio.emit('set_GPS_state', json.dumps({'gps': state}))

    @staticmethod
    def set_camera_state(state):
        logging.log(logging.INFO, f"GUI SET CAMERA STATE TO {state}")
        # trigger gui
        sio.emit('set_camera_state', json.dumps({'camera': state}))

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
        sio.emit('get_camera_custom_config', json.dumps(camera_config))

    @staticmethod
    def set_camera_easy_config(weather):
        logging.log(logging.INFO, f"GUI SET CAMERA CONFIG")
        # trigger gui
        sio.emit('get_camera_state', json.dumps({'weather': weather}))
