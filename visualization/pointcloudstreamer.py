import threading
from flask import Flask, render_template
from flask_socketio import SocketIO

class PointCloudStreamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Synchronization lock for thread-safe operations
        self.lock = threading.Lock()
        self.points_xyz = None
        self.points_bgr = None

        # Define the route for the index page
        @self.app.route('/')
        def index():
            return render_template('index.html')

        # Start a background task to emit point cloud data
        self.socketio.start_background_task(self.emit_pointcloud_loop)

    def update_pointcloud(self, points_xyz, points_bgr):
        """Update the point cloud data to be streamed."""
        with self.lock:
            self.points_xyz = points_xyz
            self.points_bgr = points_bgr

    def emit_pointcloud_loop(self):
        """Continuously emit the latest point cloud data to clients."""
        while True:
            with self.lock:
                if self.points_xyz is not None and self.points_bgr is not None:
                    # Prepare data for JSON serialization
                    data = {
                        'points': self.points_xyz.tolist(),
                        'colors': self.points_bgr.tolist()
                    }
                    # Emit data to all connected clients
                    self.socketio.emit('pointcloud', data)
            # Control the update rate
            self.socketio.sleep(0.1)

    def run(self):
        """Run the Flask app."""
        self.socketio.run(self.app, host='0.0.0.0', port=5001)
