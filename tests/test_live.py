import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PyQt5.QtCore import QObject

class TestLiveView(unittest.TestCase):
    def setUp(self):
        # Patch cv2 in the module
        self.cv2_patcher = patch('blitz.data.live.cv2')
        self.mock_cv2 = self.cv2_patcher.start()

        # Setup constants
        self.mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        self.mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        self.mock_cv2.CAP_PROP_FPS = 5
        self.mock_cv2.INTER_AREA = 1
        self.mock_cv2.COLOR_BGR2GRAY = 6
        self.mock_cv2.COLOR_BGR2RGB = 4

        # Setup default behaviors
        def resize_mock(src, dsize, interpolation=None):
            # dsize is (W, H)
            # return (H, W, C)
            if src.ndim == 3:
                return np.zeros((dsize[1], dsize[0], src.shape[2]), dtype=src.dtype)
            return np.zeros((dsize[1], dsize[0]), dtype=src.dtype)

        def cvt_mock(src, code):
            if code == 6: # GRAY
                if src.ndim == 3:
                    return src[:, :, 0]
                return src
            if code == 4: # RGB
                return src
            return src

        self.mock_cv2.resize.side_effect = resize_mock
        self.mock_cv2.cvtColor.side_effect = cvt_mock

        # Import after patching
        from blitz.data.live import CamWatcher
        self.CamWatcher = CamWatcher

    def tearDown(self):
        self.cv2_patcher.stop()

    def test_cam_watcher_lifecycle(self):
        watcher = self.CamWatcher(0, 5, 10, False, 1.0)

        mock_cap = self.mock_cv2.VideoCapture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: 100 if x in [3, 4] else 30.0

        # 3 frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) + i for i in range(3)]
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

        emitted = []
        watcher.on_next_frame.connect(lambda f: emitted.append(f))

        with patch('time.sleep'):
            watcher.start_watching()

        self.assertEqual(len(emitted), 3)
        self.assertFalse(watcher._full)
        self.assertEqual(watcher._head, 3)

        data = watcher.get_buffered_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.n_images, 3)
        self.assertEqual(data.image[0, 0, 0, 0], 0)

    def test_cam_watcher_buffer_wrap(self):
        watcher = self.CamWatcher(0, 5, 10, False, 1.0)

        mock_cap = self.mock_cv2.VideoCapture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: 100 if x in [3, 4] else 30.0

        # 7 frames
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) + i for i in range(7)]
        mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

        with patch('time.sleep'):
            watcher.start_watching()

        self.assertTrue(watcher._full)
        self.assertEqual(watcher._head, 2) # 7 % 5 = 2

        data = watcher.get_buffered_data()
        self.assertEqual(data.n_images, 5)
        # Expected frames: 2, 3, 4, 5, 6
        self.assertEqual(data.image[0, 0, 0, 0], 2)
        self.assertEqual(data.image[4, 0, 0, 0], 6)

    def test_grayscale_and_resize(self):
        watcher = self.CamWatcher(0, 5, 10, True, 0.5) # Grayscale, 50%

        mock_cap = self.mock_cv2.VideoCapture.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: 100 if x in [3, 4] else 30.0

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, frame), (False, None)]

        with patch('time.sleep'):
            watcher.start_watching()

        data = watcher.get_buffered_data()
        # Expect (1, 50, 50, 1) because ensure_4d adds channel
        self.assertEqual(data.image.shape, (1, 50, 50, 1))
        self.assertEqual(data.n_images, 1)
        self.assertEqual(data.meta[0].size, (50, 50))

if __name__ == '__main__':
    unittest.main()
