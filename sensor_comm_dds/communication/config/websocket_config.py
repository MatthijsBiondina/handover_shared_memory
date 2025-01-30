from dataclasses import dataclass


@dataclass
class WebsocketConfig:
    websocket_server_url: str = "ws://localhost:9871"
    ENABLE_WS: bool = True  # Enable WebSocket server for use with PlotJuggler. Disable if not using websocket, script
