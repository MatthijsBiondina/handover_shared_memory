import asyncio
import json
import websockets


class Websocket:
    """
    This class is for sending data to a websocket, mainly for use with PlotJuggler.
        :param websocket_server_url: Websocket URL to where data is to be sent
    """
    def __init__(self, websocket_server_url="ws://localhost:9871"):

        self.websocket_server_url = websocket_server_url
        self.websocket_async_loop = asyncio.new_event_loop()

    async def async_send_data_to_websocket(self, data):
        async with websockets.connect(self.websocket_server_url) as websocket:
            await websocket.send(json.dumps(data))

    def send_data_to_websocket(self, data):
        self.websocket_async_loop.run_until_complete(self.async_send_data_to_websocket(data))
