"""Options dock tabs that wrap onto extra rows (no hidden overflow)."""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QLayout,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class FlowLayout(QLayout):
    """Left-to-right wrap. Used so Stream / Log stay visible in a narrow dock."""

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 4) -> None:
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(spacing)
        self._items: list[QLayoutItem] = []

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_h = 0
        gap = self.spacing()
        right = rect.x() + rect.width()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + gap
            if line_h > 0 and next_x - gap > right and rect.width() > 0:
                x = rect.x()
                y += line_h + gap
                next_x = x + hint.width() + gap
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


class _TabStrip(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("optTabStrip")
        self._flow = FlowLayout(self, spacing=4)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._flow.heightForWidth(width)

    def sizeHint(self) -> QSize:  # noqa: N802
        w = max(self.width(), 120)
        return QSize(w, self.heightForWidth(w))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(80, self.heightForWidth(max(self.width(), 120)))


class OptionsTabWidget(QWidget):
    """QTabWidget stand-in: wrapping push-button tabs + stacked pages."""

    currentChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: list[QPushButton] = []
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._strip = _TabStrip(self)
        self._stack = QStackedWidget(self)
        self._stack.setObjectName("optTabPane")
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(self._strip)
        col.addWidget(self._stack, stretch=1)

    def addTab(self, widget: QWidget, name: str) -> int:
        idx = self._stack.count()
        self._stack.addWidget(widget)
        btn = QPushButton(name)
        btn.setObjectName("optTab")
        btn.setCheckable(True)
        btn.setAutoDefault(False)
        btn.setDefault(False)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        btn.setMinimumWidth(0)
        btn.setMaximumHeight(28)
        btn.clicked.connect(lambda _checked=False, i=idx: self._select(i))
        self._group.addButton(btn, idx)
        self._strip._flow.addWidget(btn)
        self._buttons.append(btn)
        if idx == 0:
            btn.setChecked(True)
        self._strip.updateGeometry()
        return idx

    def count(self) -> int:
        return self._stack.count()

    def currentIndex(self) -> int:  # noqa: N802
        return self._stack.currentIndex()

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802
        if 0 <= index < self.count():
            self._buttons[index].setChecked(True)
            self._select(index)

    def _select(self, index: int) -> None:
        if index != self._stack.currentIndex():
            self._stack.setCurrentIndex(index)
            self.currentChanged.emit(index)


# Back-compat alias if anything still imports the first name.
WrappingTabWidget = OptionsTabWidget
