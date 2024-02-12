import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PyQt5.QtWidgets import QSizePolicy, QWidget


class TimePlot(pg.PlotWidget):

    def __init__(
        self,
        parent: QWidget,
        image_viewer: pg.ImageView,
        norm_range: pg.LinearRegionItem,
        **kargs,
    ) -> None:
        super().__init__(parent, **kargs)
        self.hideAxis('left')
        self.addItem(image_viewer.timeLine)
        self.timeline = image_viewer.timeLine
        self.timeline.setMovable(False)
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
        self.norm_range = norm_range
        self.addItem(self.norm_range)
        self.norm_range.hide()
        self._accept_all_events = False

    def toggle_norm_range(self) -> None:
        if self.norm_range.isVisible():
            self.norm_range.hide()
        else:
            self.norm_range.show()

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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (event.modifiers() == Qt.KeyboardModifier.NoModifier
                and event.button() == Qt.MouseButton.LeftButton
                and not self.norm_range.isVisible()):
            x = self.plotItem.vb.mapSceneToView(  # type: ignore
                event.pos()
            ).x()
            self.image_viewer.setCurrentIndex(round(x))
            event.accept()
        elif (self.norm_range.mouseHovering
                or self.norm_range.childItems()[0].mouseHovering
                or self.norm_range.childItems()[1].mouseHovering):
            super().mousePressEvent(event)
        else:
            event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        self.image_viewer.keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        self.image_viewer.keyReleaseEvent(event)


class MeasureROI(pg.PolyLineROI):

    def __init__(self, view: pg.ViewBox) -> None:
        super().__init__([[0, 0], [0, 20], [10, 10]], closed=True)
        self.handleSize = 10
        self.sigRegionChanged.connect(self.update_labels)
        self.view = view

        self.n_px: int = 1
        self.px_in_mm: float = 1
        self.show_in_mm = False

        self.line_labels = []
        self.angle_labels = []

        self.view.addItem(self)
        self.setPen(color=(128, 128, 0, 100), width=3)
        self._visible = True
        self.toggle()

    def toggle(self) -> None:
        self._visible = not self._visible
        self.setVisible(self._visible)
        for label in self.line_labels:
            label.setVisible(self._visible)
        for label in self.angle_labels:
            label.setVisible(self._visible)
        if self._visible:
            self.update_labels()

    def update_labels(self) -> None:
        self.update_angles()
        self.update_lines()

    def update_angles(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.angle_labels) > len(positions):
            label = self.angle_labels.pop()
            self.view.removeItem(label)

        for i, (_, pos) in enumerate(positions):
            _, prev_pos = positions[(i - 1) % len(positions)]
            _, next_pos = positions[(i + 1) % len(positions)]
            angle = pg.Point(pos - next_pos).angle(
                pg.Point(pos - prev_pos)
            )
            pos = self.view.mapToView(pos)
            if i < len(self.angle_labels):
                self.angle_labels[i].setPos(pos.x(), pos.y())
                self.angle_labels[i].setText(f"{angle:.2f}°")
            else:
                angle_label = pg.TextItem(f"{angle:.2f}°")
                angle_label.setPos(pos.x(), pos.y())
                self.view.addItem(angle_label)
                self.angle_labels.append(angle_label)

    def update_lines(self) -> None:
        positions = self.getSceneHandlePositions()

        while len(self.line_labels) > len(positions):
            label = self.line_labels.pop()
            self.view.removeItem(label)

        for i, (_, pos) in enumerate(positions):
            _, next_pos = positions[(i + 1) % len(positions)]
            pos = self.view.mapToView(pos)
            next_pos = self.view.mapToView(next_pos)
            length = pg.Point(pos - next_pos).length()
            mid = (pos + next_pos) / 2
            if self.show_in_mm:
                length = length * self.px_in_mm / self.n_px
            pos = self.view.mapToView(pos)
            if i < len(self.line_labels):
                self.line_labels[i].setPos(mid.x(), mid.y())
                self.line_labels[i].setText(f"{length:.2f}")
            else:
                angle_label = pg.TextItem(f"{length:.2f}")
                angle_label.setPos(mid.x(), mid.y())
                self.view.addItem(angle_label)
                self.line_labels.append(angle_label)
