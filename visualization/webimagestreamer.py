import cv2
from threading import Thread, Lock
from flask import Flask, Response, render_template_string

from cantrips.logging.logger import get_logger

logger = get_logger()

class WebImageStreamer:
    def __init__(self, title="Image Stream", port=5000):
        self.title = title
        self.port = port
        self.app = Flask(__name__)
        self.frame = None
        self.lock = Lock()
        self.define_routes()
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.daemon = True

        # Suppress Flask startup message
        import os
        import logging

        os.environ["FLASK_ENV"] = "production"  # Ensure production mode
        os.environ["FLASK_APP"] = "webimagestreamer"  # Prevent default Flask warnings

        cli = logging.getLogger("flask.cli")
        cli.propagate = False

        logger.info(f"Flask server \"{title}\" running on port {port}.")
        
        self.server_thread.start()

    def define_routes(self):
        @self.app.route("/")
        def index():
            # Updated HTML template with centered content
            return render_template_string(
                """
                <html>
                    <head>
                        <title>{{ title }}</title>
                        <style>
                            body, html {
                                height: 100%;
                                margin: 0;
                                background-color: black; /* Set the background color to black */
                            }
                            .container {
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                flex-direction: column;
                                height: 100%;
                            }
                            h1 {
                                margin-bottom: 20px;
                                text-align: center;
                                color: white; /* Set text color to white for visibility */
                            }
                            img {
                                max-width: 100%;
                                height: auto;
                            }
                        </style>
                    </head>
                    <body>
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
            # Video streaming route
            return Response(
                self.generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def run_server(self):
        import logging
        import os

        
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        # Suppress Flask startup message
        os.environ["FLASK_ENV"] = "production"  # Ensure production mode
        os.environ["FLASK_APP"] = "webimagestreamer"  # Set a dummy app name to prevent Flask warnings
        # os.environ['WERKZEUG_RUN_MAIN'] = 'true'

        # Suppress Flask logging
        cli = logging.getLogger("flask.cli")
        cli.propagate = False
        
        self.app.run(host="0.0.0.0", port=self.port, threaded=True, use_reloader=False)

    def generate_frames(self):
        while True:
            with self.lock:
                if self.frame is None:
                    continue
                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode(".jpg", self.frame)
                frame = buffer.tobytes()
            # Yield the frame in byte format
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame
