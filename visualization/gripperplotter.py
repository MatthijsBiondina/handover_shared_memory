import numpy as np
import moderngl
import cv2
from cantrips.exceptions import ContinueException
from cantrips.logging.logger import get_logger
from cyclone.cyclone_namespace import CYCLONE_NAMESPACE
from cyclone.cyclone_participant import CycloneParticipant
from cyclone.idl.defaults.float_idl import FloatSample
from cyclone.patterns.ddsreader import DDSReader
from visualization.webimagestreamer import WebImageStreamer

logger = get_logger()


class Readers:
    def __init__(self, participant: CycloneParticipant):
        self.mag = DDSReader(
            participant,
            topic_name=CYCLONE_NAMESPACE.MAGTOUCH_PROCESSED,
            idl_dataclass=FloatSample,
        )


class GripperPlotter:
    def __init__(self, participant: CycloneParticipant):
        self.participant = participant
        self.readers = Readers(participant)
        self.web_streamer = WebImageStreamer(title="Gripper Plot", port=5007)

        self.width, self.height = 800, 600
        
        # Initialize ModernGL context
        self.ctx = moderngl.create_standalone_context()
        
        # Create framebuffer for off-screen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 3)]
        )

        # Setup shader program with modified vertex shader
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
                    // Transform y from [0,1] to [-1,1] before applying scale
                    vec2 pos = in_position;
                    pos.y = (pos.y * 2.0) - 1.0;
                    // Apply scale and offset
                    pos = pos * vec2(scale_x, scale_y) + offset;
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
            """
        )

        # Initialize data buffer
        self.history_size = 100
        self.__history = np.zeros(self.history_size)
        self.setup_buffers()


    def setup_buffers(self):
        # Create vertices for the line
        vertices = np.zeros((self.history_size, 5), dtype='f4')  # x,y,r,g,b
        vertices[:, 2:5] = (0.4, 1.0, 0.4)  # Bright green color
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, '2f 3f', 'in_position', 'in_color')]
        )

    def update_plot(self, current_value: float):
        # Bind framebuffer and set dark background
        self.fbo.use()
        self.ctx.clear(0.05, 0.05, 0.05, 1.0)  # Nearly black background

        x = np.linspace(0, 1, self.history_size)
        y = self.__history

        # Normalize y data between 0 and 1
        y_norm = (y - np.min(y)) / (np.max(y) - np.min(y)) if np.ptp(y) != 0 else np.zeros_like(y)
        
        # Scale to keep within plot bounds (0.1 to 0.9 range)
        y_norm = 1 - (0.1 + (y_norm * 0.8))  # This ensures the plot stays within bounds

        # Update vertices
        vertices = np.zeros((self.history_size, 5), dtype='f4')
        vertices[:, 0] = x
        vertices[:, 1] = y_norm
        vertices[:, 2:5] = (0.4, 1.0, 0.4)  # Bright green color

        # Update buffer
        self.vbo.write(vertices.tobytes())

        # Draw line with adjusted scale and offset
        self.prog['scale_x'].value = 1.8
        self.prog['scale_y'].value = 0.9  # Reduced y scale to prevent clipping
        self.prog['offset'].value = (-0.9, 0.0)  # Center the plot
        self.vao.render(moderngl.LINE_STRIP)

        # Read the framebuffer
        img = np.frombuffer(self.fbo.read(), dtype=np.uint8)
        img = img.reshape(self.height, self.width, 3)

        # Add current value text
        cv2.putText(
            img,
            f"Current: {current_value:.2f}",
            (10, 30),  # Position in top-left
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,  # Font scale
            (100, 255, 100),  # Bright green color
            2,  # Thickness
            cv2.LINE_AA
        )

        return img

    def run(self):
        while True:
            try:
                mag: FloatSample = self.readers.mag.take()
                if mag is None:
                    raise ContinueException

                val: float = mag.value
                
                # Update history
                self.__history = np.roll(self.__history, -1)
                self.__history[-1] = val

                # Update plot and stream image
                img = self.update_plot(val)
                self.web_streamer.update_frame(img)

            except ContinueException:
                pass
            finally:
                self.participant.sleep()


if __name__ == "__main__":
    participant = CycloneParticipant(rate_hz=60)
    node = GripperPlotter(participant)
    node.run()