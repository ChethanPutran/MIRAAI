import unittest
from camera.camera_client import CameraClient


class TestCameraClient(unittest.TestCase):
    # Start recording
    def test_start_recording(self):
        client  = CameraClient()
        res = client.start_recording()
        self.assertTrue(res["message"] == "Recording started.")

    # End Recording
    def test_end_recording(self):
        client  = CameraClient()
        res = client.end_recording()
        self.assertTrue(res["message"] == "Recording ended.")

    # Get recording
    def test_get_recording(self):
        pass

    # Start live stream
    def test_start_live_stream(self):
        pass

    # End live stream
    def test_end_live_stream(self):
        pass

    # Capture photo
    def test_capture_photo(self):
        pass
    
    # Exit
    def test_exit(self):
        pass






