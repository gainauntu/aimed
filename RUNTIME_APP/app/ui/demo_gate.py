from __future__ import annotations
from PySide6 import QtWidgets

FUTURE_TITLE = "안내"
FUTURE_BODY  = "해당 기능은 정식 버전에서 제공됩니다."

def popup_future(parent):
    QtWidgets.QMessageBox.information(parent, FUTURE_TITLE, FUTURE_BODY)

def popup_error(parent, msg: str):
    QtWidgets.QMessageBox.critical(parent, "오류", msg)
