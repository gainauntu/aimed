from __future__ import annotations
from PySide6 import QtWidgets

class PinDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, title="관리자 PIN", prompt="PIN을 입력하세요."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(380)

        v = QtWidgets.QVBoxLayout(self)
        lab = QtWidgets.QLabel(prompt)
        lab.setWordWrap(True)
        v.addWidget(lab)

        self.ed = QtWidgets.QLineEdit()
        self.ed.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.ed.setPlaceholderText("PIN")
        v.addWidget(self.ed)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        v.addWidget(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def pin(self) -> str:
        return self.ed.text().strip()
