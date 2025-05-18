import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QWidget,
    QTextEdit, QMessageBox
)

from PyQt5.QtCore import QThread, pyqtSignal

from ..core.omr_processor import LocalOMRProcessor
from ..utils.logger import setup_logger


class OMRProcessingThread(QThread):
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, processor, pdf_path):
        super().__init__()
        self.processor = processor
        self.pdf_path = pdf_path

    def run(self):
        try:
            results = self.processor.process_sheet_music(self.pdf_path)
            self.processing_complete.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))


class AudiverisConversionThread(QThread):
    conversion_complete = pyqtSignal(str)  # Path to MXL
    conversion_error = pyqtSignal(str)  # Error message

    def __init__(self, processor, image_path, output_dir, audiveris_jar_path=None):
        super().__init__()
        self.processor = processor
        self.image_path = image_path
        self.output_dir = output_dir
        self.audiveris_jar_path = audiveris_jar_path

    def run(self):
        try:
            # The omr_processor.convert_to_mxl_audiveris method handles the default JAR path if
            # self.audiveris_jar_path is None or not provided.
            mxl_file = self.processor.convert_to_mxl_audiveris(
                self.image_path,
                self.output_dir,
                audiveris_jar_path=self.audiveris_jar_path
            )
            if mxl_file:
                self.conversion_complete.emit(mxl_file)
            else:
                self.conversion_error.emit("Audiveris conversion did not return a file path. Check logs.")
        except Exception as e:
            self.conversion_error.emit(f"Error during Audiveris conversion thread: {str(e)}")


class SuperSMMMusicConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.logger = setup_logger('ui')
        self.omr_processor = LocalOMRProcessor()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('SuperSMM - Sheet Music Converter')
        self.setGeometry(100, 100, 800, 600)

        # Central Widget
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # File Selection Section
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel('No file selected')
        select_file_btn = QPushButton('Select PDF')
        select_file_btn.clicked.connect(self.select_pdf)
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(select_file_btn)

        # Process Button
        process_btn = QPushButton('Process Sheet Music')
        process_btn.clicked.connect(self.process_sheet_music)

        # Results Display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)

        # Add to main layout
        main_layout.addLayout(file_layout)
        main_layout.addWidget(process_btn)

        # Audiveris Conversion Button
        audiveris_btn = QPushButton('Convert Image to MXL (Audiveris)')
        audiveris_btn.clicked.connect(self.convert_with_audiveris)
        main_layout.addWidget(audiveris_btn)

        main_layout.addWidget(self.results_text)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def select_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open PDF File',
            '',
            'PDF Files (*.pdf)'
        )

        if file_path:
            self.file_path_label.setText(file_path)
            self.logger.info(f'Selected PDF: {file_path}')

    def process_sheet_music(self):
        file_path = self.file_path_label.text()

        if file_path == 'No file selected':
            QMessageBox.warning(
                self, 'Error', 'Please select a PDF file first.')
            return

        # Start processing in a separate thread
        self.processing_thread = OMRProcessingThread(
            self.omr_processor,
            file_path
        )
        self.processing_thread.processing_complete.connect(
            self.display_results)
        self.processing_thread.error_occurred.connect(
            self.handle_processing_error)
        self.processing_thread.start()

        # Update UI
        self.results_text.clear()
        self.results_text.append('Processing... Please wait.')

    def display_results(self, results):
        self.results_text.clear()
        self.results_text.append(f'Processed {len(results)} pages:\n')

        for i, page_result in enumerate(results, 1):
            staff_lines = page_result['staff_line_detection']
            self.results_text.append(f'Page {i}:')
            self.results_text.append(
                f'  Total Staff Lines: {staff_lines.get("total_lines", 0)}')
            self.results_text.append(
                f'  Horizontal Lines: {staff_lines.get("horizontal_lines", 0)}')
            self.results_text.append(
                f'  Staff Line Spacing: {staff_lines.get("staff_line_spacing", 0):.2f} pixels\n')

    def handle_processing_error(self, error_msg):
        QMessageBox.critical(self, 'Processing Error', error_msg)
        self.results_text.clear()
        self.results_text.append(f'Error: {error_msg}')


    def convert_with_audiveris(self):
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Image for Audiveris',
            '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)'
        )
        if not image_path:
            self.logger.info("Audiveris conversion: No image selected.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Output Directory for MusicXML'
        )
        if not output_dir:
            self.logger.info("Audiveris conversion: No output directory selected.")
            return

        self.logger.info(f"Starting Audiveris conversion for {image_path}, output to {output_dir}")
        self.results_text.setText(f"Audiveris: Converting {image_path}...")

        # The audiveris_jar_path will be handled by the LocalOMRProcessor's default if not specified here
        self.audiveris_thread = AudiverisConversionThread(
            self.omr_processor,
            image_path,
            output_dir
        )
        self.audiveris_thread.conversion_complete.connect(self.handle_audiveris_conversion_complete)
        self.audiveris_thread.conversion_error.connect(self.handle_audiveris_conversion_error)
        self.audiveris_thread.start()

    def handle_audiveris_conversion_complete(self, mxl_path):
        self.logger.info(f"Audiveris conversion successful: {mxl_path}")
        QMessageBox.information(
            self,
            'Audiveris Conversion Successful',
            f'MusicXML file saved to:\n{mxl_path}'
        )
        self.results_text.append(f"\nAudiveris: Conversion successful. MXL: {mxl_path}")

    def handle_audiveris_conversion_error(self, error_msg):
        self.logger.error(f"Audiveris conversion error: {error_msg}")
        QMessageBox.critical(
            self,
            'Audiveris Conversion Error',
            f'An error occurred:\n{error_msg}\n\nPlease ensure Audiveris and Java are correctly installed and configured (check AUDIVERIS_JAR_PATH in audiveris_converter.py or environment variable).'
        )
        self.results_text.append(f"\nAudiveris: Conversion failed. Error: {error_msg}")


def main():
    app = QApplication(sys.argv)
    window = SuperSMMMusicConverter()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
