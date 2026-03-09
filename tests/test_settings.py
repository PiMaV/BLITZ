import sys
from unittest.mock import MagicMock, patch

import pytest

@pytest.fixture(scope="module", autouse=True)
def mock_dependencies():
    """Mock heavy dependencies for the duration of this module's tests.

    This ensures that the settings module can be imported even in environments
    missing GUI/data processing libraries, while preventing mocks from leaking
    into the global sys.modules permanently.
    """
    mock_modules = [
        'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtWidgets', 'PyQt6.QtGui',
        'cv2', 'numba', 'psutil', 'pyqtgraph', 'pyqtgraph.graphicsItems',
        'pyqtgraph.graphicsItems.GradientEditorItem', 'numpy', 'natsort',
        'requests', 'socketio', 'blitz.data', 'blitz.layout'
    ]

    # Create and store original modules to restore later
    original_modules = {mod: sys.modules.get(mod) for mod in mock_modules}

    # Apply mocks
    for mod in mock_modules:
        sys.modules[mod] = MagicMock()

    yield

    # Restore original modules
    for mod, original in original_modules.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original

def test_get_project_raises_runtime_error_when_none():
    """Verify get_project raises RuntimeError when PROJECT_SETTINGS is None."""
    # Import inside the test to ensure it uses the mocks
    from blitz import settings
    with patch('blitz.settings.PROJECT_SETTINGS', None):
        with pytest.raises(RuntimeError, match="Project settings are not selected"):
            settings.get_project("any_setting")

def test_set_project_raises_runtime_error_when_none():
    """Verify set_project raises RuntimeError when PROJECT_SETTINGS is None."""
    from blitz import settings
    with patch('blitz.settings.PROJECT_SETTINGS', None):
        with pytest.raises(RuntimeError, match="Project settings are not selected"):
            settings.set_project("any_setting", "any_value")

def test_connect_sync_project_raises_runtime_error_when_none():
    """Verify connect_sync_project raises RuntimeError when PROJECT_SETTINGS is None."""
    from blitz import settings
    mock_signal = MagicMock()
    mock_getter = MagicMock()
    with patch('blitz.settings.PROJECT_SETTINGS', None):
        with pytest.raises(RuntimeError, match="Project settings are not selected"):
            settings.connect_sync_project("any_setting", mock_signal, mock_getter)
