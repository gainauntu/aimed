from __future__ import annotations
import sys
from PySide6 import QtWidgets
from app.ui.main_window import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
