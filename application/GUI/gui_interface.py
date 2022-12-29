import logging
import subprocess
from application.utils.settings import GUI_conf
from eventlet import listen as wsgi_listen
from eventlet.wsgi import server as wsgi_server
import socketio

import json

sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)


class GUIInterface:
    _connect, _disconnect, _camera_start, _camera_stop = None, None, None, None

    @staticmethod
    def start_GUI(connect, disconnect, camera_start, camera_stop):
        if not GUI_conf["GUI ON"]:
            return False

        wsgi_server(wsgi_listen('', GUI_conf["GUI server port"]), app)
        GUIInterface._connect = connect
        GUIInterface._disconnect = disconnect
        GUIInterface._camera_start = camera_start
        GUIInterface._camera_stop = camera_stop
        subprocess.Popen(GUI_conf["GUI startup script"], shell=True)

    @staticmethod
    def set_triggers(connect, disconnect, camera_start, camera_stop):
        wsgi_server(wsgi_listen(('', 5000)), app)
        GUIInterface._connect = connect
        GUIInterface._disconnect = disconnect
        GUIInterface._camera_start = camera_start
        GUIInterface._camera_stop = camera_stop

    # default events

    @staticmethod
    @sio.event
    def connect(sid, environ):
        logging.log(logging.INFO, f"GUI CONNECTION ESTABLISHED: {sid}")
        GUIInterface._connect(sid, environ)

    @staticmethod
    @sio.event
    def disconnect(sid):
        logging.log(logging.INFO, f"DISCONNECTED GUI: {sid}")
        GUIInterface._disconnect(sid)

    # custom events

    @staticmethod
    @sio.event
    def start_recording(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA START RECEIVED: {sid}")
        GUIInterface._camera_start(sid, data)

    @staticmethod
    @sio.event
    def start_view(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA START RECEIVED: {sid}")
        GUIInterface._camera_start(sid, data)

    @staticmethod
    @sio.event
    def stop_recording(sid, data):
        logging.log(logging.INFO, f"GUI CAMERA STOP RECEIVED: {sid}")
        GUIInterface._camera_stop(sid, data)

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
