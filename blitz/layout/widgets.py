import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QSizePolicy, QWidget


class TimePlot(pg.PlotWidget):

    def __init__(
        self,
        parent: QWidget,
        image_viewer: pg.ImageView,
        **kargs,
    ) -> None:
        super().__init__(parent, **kargs)
        self.hideAxis('left')
        self.addItem(image_viewer.timeLine)
        self.timeline = image_viewer.timeLine
        self.addItem(image_viewer.frameTicks)
        self.addItem(image_viewer.normRgn)
        self.old_roi_plot = image_viewer.ui.roiPlot
        self.old_roi_plot.setParent(None)
        self.image_viewer = image_viewer
        sizePolicy = QSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QSize(0, 40))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
        else:
            rotation = event.angleDelta().y()
            if rotation > 0:
                pos = self.timeline.getPos()
                self.timeline.setPos((pos[0]+1, pos[1]))
            if rotation < 0:
                pos = self.timeline.getPos()
                self.timeline.setPos((pos[0]-1, pos[1]))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)
        self.image_viewer.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        super().keyReleaseEvent(event)
        self.image_viewer.keyReleaseEvent(event)
