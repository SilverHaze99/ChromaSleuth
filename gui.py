"""
ChromaSleuth - PySide6 GUI
Modern dark-themed interface for color analysis
"""

import sys
import json
import os
import ctypes
from pathlib import Path
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QComboBox, 
                               QSpinBox, QCheckBox, QFileDialog, QScrollArea,
                               QGroupBox, QProgressBar, QMessageBox, QFrame,
                               QGridLayout, QSplitter, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import QStyle
from PIL import Image
import numpy as np

from main import ColorAnalyzer, AnalysisResult

ASSETS_PATH = Path(__file__).resolve().parent / "assets"

class AnalysisThread(QThread):
    """Background thread for image analysis"""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, analyzer, image_path, settings):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path
        self.settings = settings
    
    def run(self):
        try:
            self.progress.emit("Loading image...")
            result = self.analyzer.analyze_image(
                self.image_path,
                top_colors=self.settings['top_colors'],
                method=self.settings['method'],
                reduce_palette=self.settings['reduce_palette'],
                filter_extremes=self.settings['filter_extremes'],
                resize_large=self.settings['resize_large']
            )
            self.progress.emit("Analysis complete")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ColorCard(QFrame):
    """Widget to display a single color with information"""
    
    def __init__(self, color_data: dict, parent=None):
        super().__init__(parent)
        self.setProperty("copied", False)
        self.color_data = color_data
        self.setLayout(QHBoxLayout())
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        self.setStyleSheet("""
            ColorCard {
                background-color: #1e1e1e;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
            }
            ColorCard:hover {
                border: 1px solid #00bcd4;
                background-color: #252525;
            }
            ColorCard[copied="true"] {
                background-color: #00bcd430;
                border: 1px solid #00bcd4;
            }

            /* Styles für die Labels INNERHALB der ColorCard */
            ColorCard QLabel {
                background-color: transparent;
            }
            ColorCard QLabel#rankLabel {
                font-size: 16px; 
                font-weight: bold; 
                color: #00bcd4;
            }
            ColorCard QLabel#hexLabel {
                color: #e1e1e1; 
                font-family: 'Consolas', monospace;
            }
            ColorCard QLabel#percentageLabel {
                font-size: 18px; 
                font-weight: bold; 
                color: #e1e1e1;
            }
            ColorCard QLabel#nameLabel {
                color: #888; 
                font-size: 12px; 
                font-style: italic;
            }
            ColorCard QLabel:not(#rankLabel):not(#hexLabel):not(#percentageLabel):not(#nameLabel) {
                color: #b0b0b0; 
                font-size: 11px;
            }
        """)
        
        layout = self.layout()
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        # Color preview
        color_preview = QLabel()
        color_preview.setFixedSize(60, 60)
        rgb = self.color_data['rgb']
        color_preview.setStyleSheet(f"""
            background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]});
            border: 2px solid #00bcd4;
            border-radius: 4px;
        """)
        layout.addWidget(color_preview)
        
        # Color info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        
        rank_label = QLabel(f"#{self.color_data['rank']}")
        rank_label.setObjectName("rankLabel")
        info_layout.addWidget(rank_label)
        
        hex_label = QLabel(f"HEX: {self.color_data['hex']}")
        hex_label.setObjectName("hexLabel")
        info_layout.addWidget(hex_label)
        
        rgb_label = QLabel(f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}")
        info_layout.addWidget(rgb_label)
        
        hsl = self.color_data['hsl']
        hsl_label = QLabel(f"HSL: {hsl[0]}°, {hsl[1]}%, {hsl[2]}%")
        info_layout.addWidget(hsl_label)
        
        layout.addLayout(info_layout)
        
        # Percentage and name
        details_layout = QVBoxLayout()
        details_layout.setSpacing(4)
        
        percentage_label = QLabel(f"{self.color_data['percentage']:.2f}%")
        percentage_label.setObjectName("percentageLabel")
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        details_layout.addWidget(percentage_label)
        
        name_label = QLabel(self.color_data['color_name'])
        name_label.setObjectName("nameLabel")
        name_label.setAlignment(Qt.AlignRight)
        details_layout.addWidget(name_label)
        
        layout.addLayout(details_layout)
        
        # Make clickable for copying
        self.setCursor(Qt.PointingHandCursor)
    
    # Click to Copy
    def mousePressEvent(self, event):
        """Copy hex color to clipboard on click"""
        if event.button() == Qt.LeftButton:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.color_data['hex'])
            
            self.setProperty("copied", True)
            self.style().polish(self)
            self.update() # Wichtig!

            def reset_style():
                self.setProperty("copied", False)
                self.style().polish(self)
                self.update() # Wichtig!

            QTimer.singleShot(250, reset_style)


class DropZone(QFrame):  # <--- GEÄNDERT: Von QLabel zu QFrame
    """Drag and drop zone for images that is also clickable."""
    file_dropped = Signal(str)
    clicked = Signal()  # <--- NEU: Signal für Klicks

    def __init__(self, icon: QIcon, parent=None): # <--- GEÄNDERT: Nimmt ein Icon entgegen
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.icon = icon
        self.setup_ui()
    
    def setup_ui(self):
        # --- NEU: Layout für Icon und Text innerhalb des Frames ---
        self.setObjectName("dropZone") # Wichtig für das Styling
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(10)

        # Icon Label
        icon_label = QLabel()
        icon_label.setPixmap(self.icon.pixmap(QSize(32, 32)))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # Text Label
        text_label = QLabel("Drag & Drop Image Here\n\nOr click to browse Files")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        # --- GEÄNDERT: Stylesheet zielt jetzt auf den QFrame und seine Kinder ---
        self.setStyleSheet("""
            QFrame#dropZone {
                background-color: #1a1a1a;
                border: 3px dashed #00bcd4;
                border-radius: 12px;
                color: #b0b0b0; /* Etwas dezenterer Text */
                font-size: 14px;
                padding: 20px;
            }
            QFrame#dropZone:hover { /* Hover-Effekt für Klickbarkeit */
                background-color: #202020;
                border: 3px solid #00bcd4;
                color: #00bcd4;
            }
            /* Style für den Text innerhalb der Zone */
            QFrame#dropZone QLabel {
                background-color: transparent;
                border: none;
                color: #00bcd4;
            }
        """)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.unsetCursor()
    # --- ENDE NEU ---

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # Temporärer Style beim Drag-Over
            self.setStyleSheet("""
                QFrame#dropZone {
                    background-color: #00bcd420;
                    border: 3px solid #00bcd4;
                    border-radius: 12px;
                    padding: 20px;
                }
                QFrame#dropZone QLabel {
                    background-color: transparent;
                    border: none;
                    color: #00bcd4;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setup_ui() # Setzt den normalen Style wieder her
    
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_dropped.emit(files[0])
        self.setup_ui() # Setzt den normalen Style wieder he


class ChromaSleuthGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        if not ASSETS_PATH.exists():
            print(f"Warnung: 'assets'-Verzeichnis nicht gefunden unter {ASSETS_PATH}. Icons werden nicht geladen.")
            self.assets_available = False
        else:
            self.assets_available = True
            
        self.setWindowIcon(self.get_icon("app-icon"))

        self.analyzer = ColorAnalyzer(enable_logging=False)
        self.current_result = None
        self.current_image_path = None
        self.setup_ui()
        
    def get_icon(self, name: str) -> QIcon:
        """Lädt ein Icon aus dem Assets-Ordner. Bevorzugt SVG, nutzt PNG als Fallback."""
        if not self.assets_available:
            return QIcon()  # Leeres Icon zurückgeben, wenn keine Assets vorhanden sind
        
        for ext in ['svg', 'png']:
            path = ASSETS_PATH / f"{name}.{ext}"
            if path.exists():
                return QIcon(str(path)) # Wichtig: QIcon erwartet einen String-Pfad
                
        print(f"Warnung: Icon '{name}' nicht im Assets-Ordner gefunden.")
        return QIcon() # Leeres Icon als Fallback

    def setup_ui(self):
        self.setWindowTitle("ChromaSleuth - Color Analysis Tool")
        self.setMinimumSize(1200, 800)
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (controls and preview)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (results)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([450, 750]) # Setzt die Startbreite
        
        main_layout.addWidget(splitter)
        
    def apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                background-color: #121212;
                color: #e1e1e1;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QPushButton {
                background-color: transparent;
                color: #e2e2e2;
                border: 2px solid #00bcd4;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 13px;
                min-height: 32px;
            }
            QPushButton:hover {
                color: #00bcd4;
                background-color: #00bcd410;
            }
            QPushButton:pressed {
                background-color: #00bcd420;
            }
            QPushButton:disabled {
                background-color: transparent;
                border: 2px solid #3d3d3d;
                color: #666;
            }
            QComboBox, QSpinBox {
                background-color: #1e1e1e;
                color: #e1e1e1;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 24px;
                max-height: 28px;
            }
            QComboBox:hover, QSpinBox:hover {
                border: 1px solid #00bcd4;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 8px;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 16px;
                background-color: #2d2d2d;
                border: none;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #00bcd4;
            }
            QCheckBox {
                color: #e1e1e1;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
                border: 2px solid #2d2d2d;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #00bcd4;
            }
            QCheckBox::indicator:checked {
                background-color: #00bcd4;
                border: 1px solid #00bcd4;
            }
            QGroupBox {
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                font-weight: bold;
                color: #00bcd4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                background-color: #121212;
            }
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QProgressBar {
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                background-color: #1e1e1e;
                text-align: center;
                color: #e1e1e1;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 3px;
            }
            QLabel {
                color: #e1e1e1;
            }
        """)
    
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        panel.setMinimumWidth(400)
        
        # Scroll area for left panel
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Title
        title = QLabel("ChromaSleuth")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #00bcd4; padding: 8px 0;")
        layout.addWidget(title)
        
        subtitle = QLabel("Advanced Color Analysis")
        subtitle.setStyleSheet("font-size: 12px; color: #888; margin-bottom: 8px;")
        layout.addWidget(subtitle)
        
        # Drop zone
        self.drop_zone = DropZone(self.get_icon("folder"))
        self.drop_zone.file_dropped.connect(self.load_image)
        self.drop_zone.clicked.connect(self.browse_file) # Klick-Signal verbinden
        layout.addWidget(self.drop_zone)
        
        # Browse button
        #browse_btn = QPushButton("Browse Files")
        #browse_btn.setIcon(self.get_icon("folder"))
        #browse_btn.setIconSize(QSize(20, 20))
        #browse_btn.clicked.connect(self.browse_file)
        #layout.addWidget(browse_btn)
        
        # Settings group
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)
        settings_layout.setColumnStretch(1, 1)

        # Method selection
        method_label = QLabel("Method:")
        method_label.setStyleSheet("color: #e1e1e1; font-weight: normal;")
        settings_layout.addWidget(method_label, 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Histogram (Fast)", "Sampling", "K-means (Best Quality)"])
        self.method_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.method_combo, 0, 1)
        
        # Top colors
        top_colors_label = QLabel("Top Colors:")
        top_colors_label.setStyleSheet("color: #e1e1e1; font-weight: normal;")
        settings_layout.addWidget(top_colors_label, 1, 0)
        self.top_colors_spin = QSpinBox()
        self.top_colors_spin.setRange(3, 20)
        self.top_colors_spin.setValue(10)
        settings_layout.addWidget(self.top_colors_spin, 1, 1)
        
        # Checkboxes
        self.reduce_palette_check = QCheckBox("Reduce similar colors")
        self.reduce_palette_check.setChecked(True)
        self.reduce_palette_check.setStyleSheet("font-weight: normal;")
        settings_layout.addWidget(self.reduce_palette_check, 2, 0, 1, 2)
        
        self.filter_extremes_check = QCheckBox("Filter extreme brightness")
        self.filter_extremes_check.setStyleSheet("font-weight: normal;")
        settings_layout.addWidget(self.filter_extremes_check, 3, 0, 1, 2)
        
        self.resize_check = QCheckBox("Auto-resize large images")
        self.resize_check.setChecked(True)
        self.resize_check.setStyleSheet("font-weight: normal;")
        settings_layout.addWidget(self.resize_check, 4, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze Image")
        self.analyze_btn.setIcon(self.get_icon("analyze"))
        self.analyze_btn.setIconSize(QSize(20, 20))
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 2px solid #00bcd4;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                color: #e2e2e2;
            }
            QPushButton:hover {
                color: #00bcd4;
                background-color: #00bcd410;
            }
            QPushButton:disabled {
                border: 2px solid #3d3d3d;
                color: #666;
                background-color: transparent;
            }
        """)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.start_analysis)
        layout.addWidget(self.analyze_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 6px;")
        layout.addWidget(self.status_label)
        
        # Image preview
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setSpacing(8)
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(180)
        self.image_preview.setMaximumHeight(250)
        self.image_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.image_preview.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.image_preview.setText("No image loaded")
        self.image_preview.setScaledContents(False)
        preview_layout.addWidget(self.image_preview)
        
        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet("color: #888; font-size: 10px;")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setWordWrap(True)
        preview_layout.addWidget(self.image_info_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.setSpacing(8)
        
        self.export_json_btn = QPushButton("Export JSON")
        self.export_json_btn.setIcon(self.get_icon("json"))
        self.export_json_btn.setIconSize(QSize(20, 20))
        self.export_json_btn.setEnabled(False)
        self.export_json_btn.clicked.connect(self.export_json)
        export_layout.addWidget(self.export_json_btn)
        
        self.export_palette_btn = QPushButton("Save Palette")
        self.export_palette_btn.setIcon(self.get_icon("palette"))
        self.export_palette_btn.setIconSize(QSize(20, 20))
        self.export_palette_btn.setEnabled(False)
        self.export_palette_btn.clicked.connect(self.export_palette)
        export_layout.addWidget(self.export_palette_btn)
        
        layout.addLayout(export_layout)
        
        return scroll
    
    def create_right_panel(self):
        """Create right results panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Results header
        header_layout = QHBoxLayout()
        
        results_title = QLabel("Analysis Results")
        results_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00bcd4;")
        header_layout.addWidget(results_title)
        
        header_layout.addStretch()
        
        self.results_info = QLabel("")
        self.results_info.setStyleSheet("color: #888; font-size: 11px;")
        header_layout.addWidget(self.results_info)
        
        layout.addLayout(header_layout)
        
        # Scroll area for color cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #121212;
            }
        """)
        
        self.results_container = QWidget()
        self.results_container.setStyleSheet("background-color: #121212;")
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(12)
        self.results_layout.setContentsMargins(0, 0, 10, 0)
        
        # Placeholder
        placeholder = QLabel("No analysis performed yet\n\nLoad an image and click 'Analyze' to start")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 15px;
                padding: 60px;
                background-color: #1a1a1a;
                border: 2px dashed #2d2d2d;
                border-radius: 12px;
            }
        """)
        self.results_layout.addWidget(placeholder)
        self.results_layout.addStretch()
        
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll)
        
        return panel
    
    def browse_file(self):
        """Open file browser dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.gif);;All Files (*.*)"
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display image"""
        try:
            if not ColorAnalyzer.is_supported_format(file_path):
                QMessageBox.warning(self, "Unsupported Format", 
                                  f"File format not supported: {Path(file_path).suffix}")
                return
            
            self.current_image_path = file_path
            
            # Load and display preview
            img_pil = Image.open(file_path)
            alpha_info_text = ""
            if 'A' in img_pil.mode:
                alpha_info_text = "\n(Hinweis: Enthält Transparenz)"
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            # Create thumbnail that fits in preview area
            img_pil.thumbnail((220, 220), Image.Resampling.LANCZOS)
            
            # Convert to QPixmap
            img_array = np.array(img_pil)
            height, width, channel = img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            self.image_preview.setPixmap(pixmap)
            
            # Update info
            original_img = Image.open(file_path)
            file_size = Path(file_path).stat().st_size / 1024
            filename = Path(file_path).name
            if len(filename) > 35:
                filename = filename[:32] + "..."
            
            self.image_info_label.setText(
                f"{filename}\n"
                f"{original_img.size[0]} x {original_img.size[1]} px | {file_size:.1f} KB"
                f"{alpha_info_text}"
            )
            
            self.analyze_btn.setEnabled(True)
            self.status_label.setText("Image loaded - Ready to analyze")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def get_analysis_settings(self):
        """Get current analysis settings"""
        method_map = {
            0: 'histogram',
            1: 'sampling',
            2: 'kmeans'
        }
        
        return {
            'top_colors': self.top_colors_spin.value(),
            'method': method_map[self.method_combo.currentIndex()],
            'reduce_palette': self.reduce_palette_check.isChecked(),
            'filter_extremes': self.filter_extremes_check.isChecked(),
            'resize_large': self.resize_check.isChecked()
        }
    
    def start_analysis(self):
        """Start analysis in background thread"""
        if not self.current_image_path:
            return
        
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Analyzing...")
        
        settings = self.get_analysis_settings()
        
        self.analysis_thread = AnalysisThread(self.analyzer, self.current_image_path, settings)
        self.analysis_thread.finished.connect(self.on_analysis_finished)
        self.analysis_thread.error.connect(self.on_analysis_error)
        self.analysis_thread.progress.connect(self.status_label.setText)
        self.analysis_thread.start()
    
    def on_analysis_finished(self, result_data):
        """Handle completed analysis"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        self.current_result = AnalysisResult(result_data)
        
        status_text = f"Analysis complete - {self.current_result.processing_time:.2f}s"
        if self.current_result.had_alpha_channel:
            status_text += " (Transparency removed for analysis)"

        self.status_label.setText(status_text)
        
        self.results_info.setText(
            f"{len(self.current_result.dominant_colors)} colors | "
            f"{self.current_result.analysis_method.capitalize()} method | "
            f"{self.current_result.processing_time:.2f}s"
        )
        
        self.display_results()
        
        self.export_json_btn.setEnabled(True)
        self.export_palette_btn.setEnabled(True)
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.status_label.setText("Analysis failed")
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_msg}")
    
    def display_results(self):
        """Display analysis results"""
        # Clear previous results
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add color cards
        for i, color_data_tuple in enumerate(self.current_result.dominant_colors):
            color_data = self.current_result.get_color_data(i)
            card = ColorCard(color_data)
            self.results_layout.addWidget(card)
        
        self.results_layout.addStretch()
    
    def export_json(self):
        """Export results to JSON"""
        if not self.current_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            f"{Path(self.current_image_path).stem}_colors.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result.to_dict(), f, indent=2, ensure_ascii=False)
                
                self.status_label.setText(f"Exported to {Path(file_path).name}")
                QMessageBox.information(self, "Export Successful", 
                                      f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export:\n{str(e)}")
    
    def export_palette(self):
        """Export color palette as image"""
        if not self.current_result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Palette",
            f"{Path(self.current_image_path).stem}_palette.png",
            "PNG Files (*.png)"
        )
        
        if file_path:
            try:
                self.create_palette_image(file_path)
                self.status_label.setText(f"Palette saved as {Path(file_path).name}")
                QMessageBox.information(self, "Export Successful", 
                                      f"Palette saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save palette:\n{str(e)}")
    
    def create_palette_image(self, output_path):
        """Create and save palette image"""
        width = 800
        height = len(self.current_result.dominant_colors) * 80 + 100
        
        img = Image.new('RGB', (width, height), color=(18, 18, 18))
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font
        try:
            font_large = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        y_offset = 20
        
        for i, color_data_tuple in enumerate(self.current_result.dominant_colors):
            color_data = self.current_result.get_color_data(i)
            rgb = color_data['rgb']
            
            # Color block
            draw.rectangle([20, y_offset, 120, y_offset + 60], fill=rgb)
            
            # Text info
            text_x = 140
            draw.text((text_x, y_offset), f"#{i+1}", fill=(0, 188, 212), font=font_large)
            draw.text((text_x, y_offset + 22), color_data['hex'], fill=(225, 225, 225), font=font_small)
            draw.text((text_x + 100, y_offset + 22), 
                     f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}", 
                     fill=(176, 176, 176), font=font_small)
            draw.text((text_x + 300, y_offset + 22), 
                     f"{color_data['percentage']:.2f}%", 
                     fill=(225, 225, 225), font=font_large)
            draw.text((text_x + 400, y_offset + 22), 
                     color_data['color_name'], 
                     fill=(136, 136, 136), font=font_small)
            
            y_offset += 80
        
        img.save(output_path)


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("ChromaSleuth")
    app.setStyle("Fusion")

    if os.name == 'nt':
        try:
            myappid = 'infosecfor.chromasleuth.2.0' # random id
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except AttributeError:
            # if not supported
            print("Warning: Could not set the AppUserModelID. The taskbar icon may not be correct.")
    
    window = ChromaSleuthGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()