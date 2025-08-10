# ChromaSleuth

A powerful Python tool for extracting and analyzing dominant colors from images. This tool provides multiple analysis methods, batch processing capabilities, and comprehensive export options.

![Sample](/Sample.png)

## Features 

- **Multiple Analysis Methods**: Histogram, Pixel Sampling, and K-means clustering
- **Batch Processing**: Analyze multiple images simultaneously
- **Visual Output**: Beautiful color palette visualizations
- **Export Options**: JSON, PNG palette exports
- **Color Information**: RGB, HEX, HSL values and approximate color names
- **Performance Optimized**: Automatic image resizing and multithreading
- **CLI Interface**: Comprehensive command-line interface with extensive options

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
pip install pillow matplotlib numpy opencv-python
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

### Basic Usage

```bash
# Analyze a single image
python chromasleuth.py image.jpg

# Get top 5 colors with color names
python chromasleuth.py image.png -n 5 --color-names

# Save color palette as image
python chromasleuth.py photo.jpg -o palette.png
```

### Advanced Usage

```bash
# Use K-means clustering for better color grouping
python chromasleuth.py image.jpg -m kmeans --color-names

# Batch process multiple images
python chromasleuth.py *.jpg --batch -j results.json

# Filter extreme colors and export to JSON
python chromasleuth.py logo.png --filter-extremes -j colors.json
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
======================================================================
COLOR ANALYSIS: example_image.jpg
======================================================================
Rank Hex Code  RGB             Percentage Count      Color Name
----------------------------------------------------------------------
1    #2c5f2d   ( 44, 95, 45)    15.2%     12,456    Green
2    #87ceeb   (135,206,235)    12.8%     10,234    Cyan
3    #f4a460   (244,164, 96)     8.9%      7,123    Orange
```

### JSON Export
```json
{
  "timestamp": "2024-08-09 15:30:45",
  "image_info": {
    "filename": "example.jpg",
    "original_size": [1920, 1080],
    "analysis_method": "histogram",
    "total_unique_colors": 45678
  },
  "colors": [
    {
      "rank": 1,
      "rgb": [44, 95, 45],
      "hex": "#2c5f2d",
      "hsl": [122, 36, 27],
      "color_name": "Green",
      "count": 12456,
      "percentage": 15.2
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

## Performance Tips 

### For Large Images
```bash
# Images are automatically resized to 1920px max dimension
# To disable: use --no-resize

python chromasleuth.py large_image.jpg --no-resize -v
```

### For Many Colors
```bash
# Color reduction groups similar colors (enabled by default)
# To disable: use --no-reduce

python chromasleuth.py detailed_image.jpg --no-reduce
```

### Batch Processing
```bash
# Process all images in a directory
python chromasleuth.py /path/to/images/*.jpg --batch -j results.json

# The tool uses multithreading (4 workers) for batch processing
```

## Examples 

### Example 1: Logo Color Analysis
```bash
python chromasleuth.py logo.png -n 3 --color-names -o logo_palette.png
```
Perfect for extracting brand colors from logos.

### Example 2: Art Analysis
```bash
python chromasleuth.py artwork.jpg -m kmeans -n 8 --filter-extremes
```
Great for analyzing paintings or digital art with better color grouping.

### Example 3: Website Color Scheme
```bash
python chromasleuth.py screenshot.png -n 5 -j website_colors.json
```
Extract color schemes from website screenshots.

### Example 4: Batch Product Analysis
```bash
python chromasleuth.py product_photos/*.jpg --batch --color-names -j product_colors.json
```
Analyze multiple product photos for consistent color reporting.

## Troubleshooting 

### Common Issues

**"No module named 'cv2'"**
```bash
pip install opencv-python
```

**"Image file not found"**
- Check file path and permissions
- Ensure the image format is supported

**"Memory error with large images"**
```bash
# Use sampling method for very large images
python chromasleuth.py huge_image.jpg -m sampling --sample-factor 20
```

**"Too many colors found"**
```bash
# Use color reduction (enabled by default) or filtering
python chromasleuth.py image.jpg --filter-extremes
```

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
- Web interface
- More color spaces (LAB, XYZ)
- Color harmony analysis
- Batch comparison tools
- Integration with design tools
- Integration of webcolors or colour for exact CSS/X11 color names.
- GUI? Maybe...with Drag&Drop (Tkinter or more fancy)
- Docker support maybe, but idk...
- Benchmark Mode cause why not^^
- “Color Emotion” analysis
- Better UI cause right now matplotlib.pyplot is not that good

## License 

This tool is provided as-is for OSINT research and analysis purposes. Users are responsible for ensuring compliance with applicable laws and regulations in their jurisdiction.
This tool is provided under the MIT-License

## Changelog 

### Version 2.0 (Enhanced)
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
