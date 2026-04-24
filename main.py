import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from image_processing import (
    ImageProcessingError,
    apply_convolution,
    apply_histogram_specification,
    create_histogram_preview,
)


class LeafProcessingApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Peningkatan Mutu Citra")
        self.resize(1300, 760)

        self.original_image = None
        self.reference_image = None
        self.processed_image = None

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        controls = self._build_controls()
        viewer = self._build_viewer()

        root_layout.addWidget(controls, 1)
        root_layout.addWidget(viewer, 2)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignTop)

        io_group = QGroupBox("Input / Output")
        io_layout = QVBoxLayout(io_group)

        self.btn_load_input = QPushButton("Upload Gambar")
        self.btn_load_input.clicked.connect(self.load_input_image)
        io_layout.addWidget(self.btn_load_input)

        self.btn_load_reference = QPushButton("Upload Gambar Referensi Histogram")
        self.btn_load_reference.clicked.connect(self.load_reference_image)
        io_layout.addWidget(self.btn_load_reference)

        self.btn_save = QPushButton("Simpan Hasil")
        self.btn_save.clicked.connect(self.save_image)
        io_layout.addWidget(self.btn_save)

        filter_group = QGroupBox("Convolution (Mask Processing)")
        filter_layout = QGridLayout(filter_group)

        filter_layout.addWidget(QLabel("Pilih Filter"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Smoothing", "Sharpening", "Sobel Edge", "Laplacian Edge"])
        self.filter_combo.currentTextChanged.connect(self.process_image)
        filter_layout.addWidget(self.filter_combo, 0, 1)

        self.kernel_label = QLabel("Kernel Size: 3")
        filter_layout.addWidget(self.kernel_label, 1, 0, 1, 2)
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(3)
        self.kernel_slider.valueChanged.connect(self._on_kernel_change)
        filter_layout.addWidget(self.kernel_slider, 2, 0, 1, 2)

        self.strength_label = QLabel("Sharpen Strength: 5")
        filter_layout.addWidget(self.strength_label, 3, 0, 1, 2)
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(0, 30)
        self.strength_slider.setValue(5)
        self.strength_slider.valueChanged.connect(self._on_strength_change)
        filter_layout.addWidget(self.strength_slider, 4, 0, 1, 2)

        hist_group = QGroupBox("Histogram Specification")
        hist_layout = QVBoxLayout(hist_group)
        self.hist_toggle = QComboBox()
        self.hist_toggle.addItems(["Off", "On"])
        self.hist_toggle.currentTextChanged.connect(self.process_image)
        hist_layout.addWidget(QLabel("Aktifkan Histogram Specification"))
        hist_layout.addWidget(self.hist_toggle)
        hist_layout.addWidget(QLabel("Mode Tampilan Histogram"))
        self.hist_mode_combo = QComboBox()
        self.hist_mode_combo.addItems(["RGB", "Grayscale"])
        self.hist_mode_combo.currentTextChanged.connect(self.process_image)
        hist_layout.addWidget(self.hist_mode_combo)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_controls)

        layout.addWidget(io_group)
        layout.addWidget(filter_group)
        layout.addWidget(hist_group)
        layout.addWidget(self.btn_reset)

        return panel

    def _build_viewer(self) -> QWidget:
        panel = QWidget()
        layout = QHBoxLayout(panel)

        before_layout = QVBoxLayout()
        after_layout = QVBoxLayout()

        before_layout.addWidget(QLabel("Sebelum"))
        self.before_view = QLabel("Belum ada gambar")
        self.before_view.setAlignment(Qt.AlignCenter)
        self.before_view.setStyleSheet("border: 1px solid #777; min-height: 420px;")
        before_layout.addWidget(self.before_view)
        before_layout.addWidget(QLabel("Histogram Sebelum"))
        self.before_hist_view = QLabel("Histogram belum tersedia")
        self.before_hist_view.setAlignment(Qt.AlignCenter)
        self.before_hist_view.setStyleSheet("border: 1px solid #777; min-height: 180px;")
        before_layout.addWidget(self.before_hist_view)

        after_layout.addWidget(QLabel("Sesudah"))
        self.after_view = QLabel("Belum ada hasil")
        self.after_view.setAlignment(Qt.AlignCenter)
        self.after_view.setStyleSheet("border: 1px solid #777; min-height: 420px;")
        after_layout.addWidget(self.after_view)
        after_layout.addWidget(QLabel("Histogram Sesudah"))
        self.after_hist_view = QLabel("Histogram belum tersedia")
        self.after_hist_view.setAlignment(Qt.AlignCenter)
        self.after_hist_view.setStyleSheet("border: 1px solid #777; min-height: 180px;")
        after_layout.addWidget(self.after_hist_view)

        layout.addLayout(before_layout)
        layout.addLayout(after_layout)
        return panel

    def _on_kernel_change(self, value: int) -> None:
        self.kernel_label.setText(f"Kernel Size: {value}")
        self.process_image()

    def _on_strength_change(self, value: int) -> None:
        self.strength_label.setText(f"Sharpen Strength: {value}")
        self.process_image()

    def load_input_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih gambar daun",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            self._show_error("Gagal membaca gambar input.")
            return

        self.original_image = image
        self._show_image(self.before_view, self.original_image)
        self._show_histogram(self.before_hist_view, self.original_image)
        self.process_image()

    def load_reference_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih gambar referensi histogram",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            self._show_error("Gagal membaca gambar referensi.")
            return

        self.reference_image = image
        self.process_image()

    def process_image(self) -> None:
        if self.original_image is None:
            return

        self._show_image(self.before_view, self.original_image)
        self._show_histogram(self.before_hist_view, self.original_image)
        working = self.original_image.copy()

        filter_name = self.filter_combo.currentText()
        kernel_size = self.kernel_slider.value()
        strength = self.strength_slider.value()

        try:
            working = apply_convolution(working, filter_name, kernel_size, strength)

            if self.hist_toggle.currentText() == "On":
                if self.reference_image is None:
                    self._show_warning("Histogram specification aktif, tapi gambar referensi belum di-upload.")
                else:
                    working = apply_histogram_specification(working, self.reference_image)

            if self.hist_mode_combo.currentText() == "Grayscale":
                gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
                working = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            self.processed_image = working
            self._show_image(self.after_view, self.processed_image)
            self._show_histogram(self.after_hist_view, self.processed_image)
        except ImageProcessingError as exc:
            self._show_error(str(exc))
        except Exception as exc:
            self._show_error(f"Terjadi error saat memproses gambar: {exc}")

    def save_image(self) -> None:
        if self.processed_image is None:
            self._show_warning("Belum ada hasil untuk disimpan.")
            return

        default_path = str(Path.home() / "hasil_pengolahan_daun.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Simpan gambar hasil",
            default_path,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not file_path:
            return

        ok = cv2.imwrite(file_path, self.processed_image)
        if not ok:
            self._show_error("Gagal menyimpan gambar.")
            return
        self.statusBar().showMessage(f"Hasil disimpan di: {file_path}", 5000)

    def reset_controls(self) -> None:
        self.filter_combo.setCurrentText("None")
        self.hist_toggle.setCurrentText("Off")
        self.hist_mode_combo.setCurrentText("RGB")
        self.kernel_slider.setValue(3)
        self.strength_slider.setValue(5)
        self.processed_image = self.original_image.copy() if self.original_image is not None else None
        if self.processed_image is not None:
            self._show_image(self.after_view, self.processed_image)
            self._show_histogram(self.after_hist_view, self.processed_image)
        else:
            self.after_view.setText("Belum ada hasil")
            self.after_view.setPixmap(QPixmap())
            self.after_hist_view.setText("Histogram belum tersedia")
            self.after_hist_view.setPixmap(QPixmap())

    def _show_image(self, target: QLabel, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target.setPixmap(scaled)

    def _show_histogram(self, target: QLabel, image_bgr: np.ndarray) -> None:
        hist_mode = self.hist_mode_combo.currentText() if hasattr(self, "hist_mode_combo") else "RGB"
        hist_image = create_histogram_preview(image_bgr, mode=hist_mode, width=460, height=220)
        self._show_image(target, hist_image)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.original_image is not None:
            self._show_image(self.before_view, self.original_image)
            self._show_histogram(self.before_hist_view, self.original_image)
        if self.processed_image is not None:
            self._show_image(self.after_view, self.processed_image)
            self._show_histogram(self.after_hist_view, self.processed_image)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)

    def _show_warning(self, message: str) -> None:
        QMessageBox.warning(self, "Warning", message)


def main() -> None:
    app = QApplication(sys.argv)
    window = LeafProcessingApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
