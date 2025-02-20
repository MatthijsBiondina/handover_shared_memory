import inspect
from pathlib import Path
import cv2
import time
import json
import os
import logging

from threading import Thread, Lock
from flask import Flask, Response, render_template_string, jsonify
from cantrips.logging.logger import get_logger

logger = get_logger()


class WebImageStreamer:
    SAVE_ROOT = Path("/home/matt/Videos")

    def __init__(self, title="Image Stream", port=5000, save=True):
        self.title = os.path.basename(
            inspect.getfile(inspect.currentframe().f_back)
        ).replace(".py", "")
        self.save = save
        self.port = port
        self.app = Flask(__name__)
        self.frame = None
        self.lock = Lock()
        self.last_frame_time = time.time()
        self.fps = 0
        self.define_routes()
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.daemon = True

        # Suppress Flask startup message

        os.environ["FLASK_ENV"] = "production"
        os.environ["FLASK_APP"] = "webimagestreamer"

        cli = logging.getLogger("flask.cli")
        cli.propagate = False

        self.root = self.init_filesystem()

        logger.info(f'Flask server "{title}" running on port {port}.')

        self.server_thread.start()

    def init_filesystem(self):
        if not self.save:
            return None

        os.makedirs(self.SAVE_ROOT / self.title, exist_ok=True)
        ii = 0
        while True:
            folder = self.SAVE_ROOT / self.title / str(ii).zfill(2)
            if os.path.exists(folder):
                ii += 1
            else:
                break
        os.makedirs(folder)
        return folder

    def define_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(
                """
                <html>
                    <head>
                        <title>{{ title }}</title>
                        <style>
                            body, html {
                                height: 100%;
                                margin: 0;
                                background-color: black;
                                color: white;
                                font-family: Arial, sans-serif;
                            }
                            .container {
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                flex-direction: column;
                                height: 100%;
                                position: relative;
                            }
                            .fps-display {
                                position: fixed;
                                top: 20px;
                                left: 20px;
                                background-color: rgba(0, 0, 0, 0.7);
                                padding: 10px 15px;
                                border-radius: 5px;
                                font-size: 16px;
                                z-index: 1000;
                            }
                            img {
                                max-width: 100%;
                                height: auto;
                            }
                        </style>
                        <script>
                            function updateFPS() {
                                fetch('/get_fps')
                                    .then(response => response.json())
                                    .then(data => {
                                        document.getElementById('fps-counter').textContent = 
                                            `FPS: ${data.fps.toFixed(1)}`;
                                    })
                                    .catch(console.error);
                            }
                            
                            // Update FPS every 500ms
                            setInterval(updateFPS, 500);
                        </script>
                    </head>
                    <body>
                        <div class="fps-display">
                            <span id="fps-counter">FPS: 0.0</span>
                        </div>
                        <div class="container">
                            <img src="{{ url_for('video_feed') }}">
                        </div>
                    </body>
                </html>
                """,
                title=self.title,
            )

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self.generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/get_fps")
        def get_fps():
            return jsonify({"fps": self.fps})

    def run_server(self):
        import logging
        import os

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        os.environ["FLASK_ENV"] = "production"
        os.environ["FLASK_APP"] = "webimagestreamer"

        cli = logging.getLogger("flask.cli")
        cli.propagate = False

        self.app.run(host="0.0.0.0", port=self.port, threaded=True, use_reloader=False)

    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.fps = 1.0 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time
        return self.fps

    def generate_frames(self):
        while True:
            with self.lock:
                if self.frame is None:
                    continue

                # Calculate FPS
                # self.calculate_fps()

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode(".jpg", self.frame)
                frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def update_frame(self, frame):
        with self.lock:
            self.calculate_fps()
            self.frame = frame
            if self.save:
                cv2.imwrite(str(self.root / f"{time.time():.2f}.jpg"), frame)

