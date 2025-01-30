import numpy as np
import moderngl
from OpenGL.GL import *
from cantrips.configs import load_config
from cantrips.exceptions import ContinueException
from cyclone.cyclone_participant import CycloneParticipant
from cantrips.logging.logger import get_logger
from cyclone.patterns.ddsreader import DDSReader
from sensor_comm_dds.communication.data_classes.magtouch4 import MagTouch4
from visualization.webimagestreamer import WebImageStreamer
import cv2

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.mag0 = DDSReader(
            participant, topic_name="MagTouchRaw0", idl_dataclass=MagTouch4
        )


class ModernGLPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.config = load_config()
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(title="MagTouch", port=5010)

        self.width, self.height = 1200, 800

        # Initialize standalone ModernGL context (no window)
        self.ctx = moderngl.create_standalone_context()

        # Create framebuffer for off-screen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 3)]
        )

        # Setup the vertex shader for line rendering
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                in vec3 in_color;
                out vec3 color;
                uniform float scale_x;
                uniform float scale_y;
                uniform vec2 offset;
                void main() {
                    vec2 pos = in_position * vec2(scale_x, scale_y) + offset;
                    gl_Position = vec4(pos, 0.0, 1.0);
                    color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 color;
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(color, 1.0);
                }
            """,
        )

        # Initialize data buffers
        self.history_size = 100
        self.__history = np.zeros((self.history_size, 12))
        self.setup_buffers()

        # Y-axis parameters
        self.y_min, self.y_max = -75000, 10
        self.num_y_ticks = 5

    def setup_buffers(self):
        self.vbos = []
        self.vaos = []

        # Create 4 subplots (one for each taxel)
        # Updated colors for dark mode - brighter and more saturated
        colors = [
            (1.0, 0.4, 0.4),  # Bright red
            (0.4, 1.0, 0.4),  # Bright green
            (0.4, 0.4, 1.0),  # Bright blue
        ]

        for taxel in range(4):
            taxel_vbos = []
            taxel_vaos = []

            for dim in range(3):
                # Create vertices for each line
                vertices = np.zeros((self.history_size, 5), dtype="f4")  # x,y,r,g,b
                vertices[:, 2:5] = colors[dim]  # Set color

                vbo = self.ctx.buffer(vertices.tobytes())
                vao = self.ctx.vertex_array(
                    self.prog, [(vbo, "2f 3f", "in_position", "in_color")]
                )

                taxel_vbos.append(vbo)
                taxel_vaos.append(vao)

            self.vbos.append(taxel_vbos)
            self.vaos.append(taxel_vaos)

        # Setup y-axis buffer
        y_axis_vertices = np.array(
            [[0, 0, 0.3, 0.3, 0.3], [0, 1, 0.3, 0.3, 0.3]],  # Darker gray color
            dtype="f4",
        )

        self.y_axis_vbo = self.ctx.buffer(y_axis_vertices.tobytes())
        self.y_axis_vao = self.ctx.vertex_array(
            self.prog, [(self.y_axis_vbo, "2f 3f", "in_position", "in_color")]
        )

    def draw_y_axis_labels(
        self, img, offset_x, offset_y, subplot_width, subplot_height
    ):
        # Convert normalized coordinates to pixel coordinates
        pixel_x = int((offset_x + 1) * self.width / 2)

        # Calculate vertical positioning based on subplot position
        if offset_y > -0.5:  # Top subplots
            base_y = self.height / 4
            y_direction = 1
        else:  # Bottom subplots
            base_y = 3 * self.height / 4
            y_direction = 1

        y_values = np.linspace(self.y_min, self.y_max, self.num_y_ticks)
        for y_val in y_values:
            # Convert y value to normalized coordinates
            y_norm = (y_val - self.y_min) / (self.y_max - self.y_min)
            pixel_y = int(base_y + y_direction * (y_norm - 0.5) * self.height / 2)

            # Draw tick label
            label = f"{int(y_val)}"
            cv2.putText(
                img,
                label,
                (pixel_x - 45, pixel_y),  # Position the text closer to the axis
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # Font scale
                (100, 100, 100),  # Darker gray color
                1,  # Thickness
                cv2.LINE_AA,
            )

    def update_plot(self):
        # Bind our framebuffer and set dark background
        self.fbo.use()
        self.ctx.clear(0.05, 0.05, 0.05, 1.0)  # Nearly black background

        # Calculate subplot positions
        subplot_width = 1.8
        subplot_height = 1.8
        positions = [
            (-0.9, 0.1),  # bottom left
            (0.1, 0.1),  # bottom right
            (-0.9, -0.9),  # top left
            (0.1, -0.9),  # top right
        ]

        x = np.linspace(0, 1, self.history_size)

        for taxel in range(4):
            offset_x, offset_y = positions[taxel]

            # Draw y-axis for this subplot
            self.prog["scale_x"].value = subplot_width / 2
            self.prog["scale_y"].value = subplot_height / 2
            self.prog["offset"].value = (offset_x, offset_y)
            self.y_axis_vao.render(moderngl.LINES)

            for dim in range(3):
                # Update data for this line
                data_idx = taxel * 3 + dim
                y = self.__history[:, data_idx]

                # Normalize y data
                # y_norm = (y - self.y_min) / (self.y_max - self.y_min)
                y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

                # Create vertices with updated colors for dark mode
                vertices = np.zeros((self.history_size, 5), dtype="f4")
                vertices[:, 0] = x
                vertices[:, 1] = y_norm
                colors = [
                    (1.0, 0.4, 0.4),  # Bright red
                    (0.4, 1.0, 0.4),  # Bright green
                    (0.4, 0.4, 1.0),  # Bright blue
                ]
                vertices[:, 2:5] = colors[dim]

                # Update buffer
                self.vbos[taxel][dim].write(vertices.tobytes())

                # Draw line
                self.prog["scale_x"].value = subplot_width / 2
                self.prog["scale_y"].value = subplot_height / 2
                self.prog["offset"].value = (offset_x, offset_y)
                self.vaos[taxel][dim].render(moderngl.LINE_STRIP)

        # Read the framebuffer
        img = np.frombuffer(self.fbo.read(), dtype=np.uint8)
        img = img.reshape(self.height, self.width, 3)

        # Add y-axis labels to each subplot
        for offset_x, offset_y in positions:
            self.draw_y_axis_labels(
                img, offset_x, offset_y, subplot_width, subplot_height
            )

        return img

    def run(self):
        while True:
            try:
                mag = self.readers.mag0.take()
                if mag is None:
                    raise ContinueException
                try:
                    reading = [
                        val for mtt in mag.taxels for val in (mtt.x, mtt.y, mtt.z)
                    ]
                except AttributeError:
                    raise ContinueException
                self.__history = np.roll(self.__history, -1, axis=0)
                self.__history[-1] = np.array(reading)

                # Update the plot and stream the image
                img = self.update_plot()
                self.web_streamer.update_frame(img)

            except ContinueException:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=60)
    node = ModernGLPlotter(participant)
    node.run()
