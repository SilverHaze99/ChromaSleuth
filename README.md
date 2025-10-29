# ChromaSleuth

A powerful and modern tool for color analysis of images, featuring both an intuitive graphical user interface (GUI) and a scriptable command line interface (CLI).

![Sample](/Sample.png)

## Features 

- **Modern, intuitive GUI**: An attractive and user-friendly interface created with PySide6.
- **Drag & drop and click functionality**: Simply drag and drop or click images into the loading zone.
- **Interactive results**: Click on a color swatch to copy the HEX color code directly to the clipboard.
- **Powerful backend logic**: Choose between three analysis methods (histogram, sampling, K-means) to balance speed and accuracy.
- **Powerful CLI**: A fully functional command line for batch processing and automation.
- **Comprehensive export options**: Export analysis results as a JSON file or save a visual color palette as a PNG image.
- **Alpha channel detection**: The tool detects images with transparency (alpha channel) and provides the user with appropriate feedback.


### Setup

1.  **Clone the repository** (if you are using Git):
    ```bash
    git clone https://github.com/SilverHaze99/ChromaSleuth
    cd ChromaSleuth
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
4.  **Install Required Python Libraries:** Install the necessary libraries using pip within the activated virtual environment:
```bash
pip install pillow matplotlib numpy opencv-python pyside6
```

### System Requirements

- Python 3.7 or higher
- Windows, macOS, or Linux
- Minimum 4GB RAM (8GB recommended for batch processing)

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)
- GIF (.gif)

## Quick Start 
ChromaSleuth offers two interfaces: a graphical user interface (GUI) and a command line interface (CLI).

### GUI Usage

The GUI is ideal for analyzing individual images and providing a visual, interactive experience.

```bash
# Analyze a single image
python gui.py
```
### GUI-Features:
- Drag an image into the drop zone or click on it to select a file.
- Adjust the analysis settings in the left panel.
- Click “Analyze Image” to extract the dominant colors.
- Click on any color swatch in the results to copy the HEX code.
- Export the results as JSON or as a visual palette.

### CLI Usage

The CLI is perfect for automation, processing multiple files (batch mode), and integration into scripts.

```bash
# Display help to see all options
python cli.py -h

# Simple analysis of an image
python cli.py “path/to/image.jpg”

# Analysis with K-Means, output of the top 5 colors, and saving of the palette
python cli.py “logo.png” -n 5 -m kmeans --color-names -o “palette.png”

# Process all PNG files in a folder and save a global JSON
python cli.py “path/to/folder/*.png” --batch -j “all_results.json”
```

## Command Line Options 

### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `image_path` | Path(s) to image file(s) | Required |
| `-n, --top-colors` | Number of top colors to extract | 10 |
| `-h, --help` | Show help message | - |

### Analysis Methods

| Option | Description | Use Case |
|--------|-------------|----------|
| `-m histogram` | Fast histogram-based analysis | General purpose (default) |
| `-m sampling` | Pixel sampling method | Large images, custom sampling |
| `-m kmeans` | K-means clustering | Better color grouping |

### Output Options

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Save color palette as image |
| `-j, --json FILE` | Export results to JSON |
| `--color-names` | Show approximate color names |
| `--no-display` | Don't show color palette |

### Processing Options

| Option | Description |
|--------|-------------|
| `--batch` | Process multiple images |
| `--no-reduce` | Don't apply color reduction |
| `--filter-extremes` | Filter very dark/bright colors |
| `--no-resize` | Don't resize large images |
| `--sample-factor N` | Sampling factor (sampling method) |
| `-v, --verbose` | Enable verbose logging |

## Analysis Methods Explained 

### 1. Histogram Analysis (Default)
- **Speed**:  3/3 Very Fast
- **Accuracy**: 3/4 Good
- **Use Case**: General purpose, fast analysis
- **How it works**: Uses numpy to count all unique RGB values

### 2. Pixel Sampling
- **Speed**: 2/3 Fast
- **Accuracy**: 2/2 Fair
- **Use Case**: Very large images, when speed is critical
- **How it works**: Analyzes every n-th pixel (configurable)

### 3. K-means Clustering
- **Speed**: 1/3 Moderate
- **Accuracy**: 4/4 Excellent
- **Use Case**: When you need the best color grouping
- **How it works**: Groups similar colors using machine learning

## Output Formats 

### Console Output
```
Analyzing: test.png

======================================================================
                       COLOR ANALYSIS: test.png
======================================================================

Image Size:    1920 x 1080 px
Tranyparency: Yes (Flattened for analysis)
Method:        Histogram
Processing:    2.29 seconds
Total Colors:  17,014
After Filter:  47

Rank  Hex       RGB               Percentage  Count
----------------------------------------------------------------------
1     #000000   (  0,  0,  0)      38.79%      804,420
2     #002020   (  0, 32, 32)      11.53%      239,126
3     #004040   (  0, 64, 64)       8.26%      171,194
```

### JSON Export
```json
{
  "timestamp": "2025-10-29 13:12:08",
  "image_info": {
    "filename": "ChromaSleuth PR.png",
    "filepath": "G:\\Server\\ChromaSleuth PR.png",
    "original_size": [
      1920,
      1080
    ],
    "analysis_method": "kmeans",
    "processing_time": 5.62,
    "had_alpha_channel": true
  },
  "colors": [
    {
      "rank": 1,
      "rgb": [
        4,
        4,
        6
      ],
      "hex": "#040406",
      "hsl": [
        240,
        20,
        1
      ],
      "color_name": "Very Dark",
      "count": 445666,
      "percentage": 21.492380401234566
    }
  ]
}
```

### Visual Palette
The tool generates beautiful visualizations with:
- Horizontal bar chart showing color percentages
- Color swatches below the chart
- Smart text color (black/white) based on background brightness
- Optional color names in the visualization

  ![example](/example.png)

## Troubleshooting 

### Performance Issues

- **Large images**: Use `--sample-factor` with higher values
- **Many images**: Batch processing uses 4 threads by default
- **Memory usage**: Images are automatically resized unless `--no-resize` is used

## Technical Details 

### Color Space Conversions
- **RGB**: Native format, used for all calculations
- **HEX**: Standard web format (#RRGGBB)
- **HSL**: Used for color naming and filtering

### Color Naming Algorithm
The tool uses a simple HSL-based algorithm to provide approximate color names:
- Hue ranges determine basic colors (Red, Orange, Yellow, etc.)
- Saturation and Lightness determine variants (Dark, Light, Gray)

### Performance Optimizations
- Automatic image resizing for large images (>1920px)
- Numpy vectorization for histogram analysis
- Multithreading for batch processing
- Smart memory management

## Contributing 

Feel free to contribute improvements:

1. **Bug Reports**: Include sample images and error messages
2. **Feature Requests**: Describe the use case and expected behavior
3. **Code Contributions**: Follow the existing code style

### Ideas for Future Features
- ~~GUI? Maybe...with Drag&Drop (Tkinter or more fancy)~~ DONE✅
- ~~Better UI cause right now matplotlib.pyplot is not that good~~ DONE✅
- Web interface as an alternative to the desktop GUI
- Integration of webcolors or colour for exact CSS/X11 color names.
- Analysis of color harmonies (complementary, analogous, etc.).
- More color spaces (LAB, XYZ)
- Batch comparison tools
- Integration with design tools
- “Color Emotion” analysis
- Live-Masking of the color in the image


## License 

This tool is provided as-is for OSINT research and analysis purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.
This tool is provided under the MIT-License
Icons used are provided by "lucide" [lucide-license](https://github.com/lucide-icons/lucide/blob/main/LICENSE)

## Changelog 

### Version 3.0 (GUI Edition)
- ✅ Added: Full graphical user interface (GUI) built with PySide6 for a modern and responsive experience.
- ✅ Revised: Drag & drop zone is now clickable, replacing the separate “Browse” button for a cleaner UI.
- ✅ Added: “Click-to-copy” functionality for all color result cards in the GUI.
- ✅ Refactored: Centralized and optimized stylesheets to enable visual feedback (e.g., when copying) and improve maintainability.
- ✅ Added: Alpha channel (transparency) detection with clear feedback for the user in GUI and CLI.
- ✅ Added: Icons for app windows and all main buttons to improve user guidance.

### Version 2.0 (Enhanced CLI)
- ✅ Added K-means clustering method
- ✅ Batch processing with multithreading
- ✅ JSON export functionality
- ✅ Color naming system
- ✅ Improved CLI with comprehensive options
- ✅ Better error handling and logging
- ✅ Performance optimizations
- ✅ Enhanced visualizations

### Version 1.0 (Original; not on Github, was just local and really simple... )
- ✅ Basic histogram analysis
- ✅ Pixel sampling method
- ✅ Simple CLI interface
- ✅ Matplotlib visualizations
