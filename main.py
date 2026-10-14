import sys
from PyQt6.QtWidgets import QApplication
from ui import HopfieldApp


def main() -> None:
    app = QApplication(sys.argv)
    window = HopfieldApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()