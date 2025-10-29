import time
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
import cv2


class ColorAnalyzer:
    """
    Core color analysis engine
    Handles all image processing and color extraction
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    def __init__(self, sample_factor: int = 10, enable_logging: bool = False):
        """
        Initialize the ColorAnalyzer
        
        Args:
            sample_factor: Analyze every n-th pixel (performance optimization)
            enable_logging: Enable detailed logging
        """
        self.sample_factor = sample_factor
        self._setup_logging(enable_logging)
        
    def _setup_logging(self, enable: bool) -> None:
        """Setup logging configuration"""
        if enable:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def is_supported_format(file_path: Union[str, Path]) -> bool:
        """Check if the file format is supported"""
        return Path(file_path).suffix.lower() in ColorAnalyzer.SUPPORTED_FORMATS
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image and convert it to RGB, and detect alpha channel
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (PIL Image object in RGB mode, boolean indicating if alpha was present)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not self.is_supported_format(image_path):
            raise ValueError(f"Unsupported format: {image_path.suffix}")
        
        try:
            img = Image.open(image_path)
            had_alpha = 'A' in img.mode

            if img.mode != 'RGB':
                self.logger.info(f"Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            return img, had_alpha
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    def resize_image_if_large(self, image: Image.Image, max_size: int = 1920) -> Tuple[Image.Image, bool]:
        """
        Resize image if it's larger than max_size to improve performance
        
        Args:
            image: PIL Image object
            max_size: Maximum width/height in pixels
            
        Returns:
            Tuple of (resized image, was_resized boolean)
        """
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            self.logger.info(f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]}")
            return image.resize(new_size, Image.Resampling.LANCZOS), True
        return image, False
    
    def extract_colors_histogram(self, image: Image.Image) -> Dict[Tuple[int, int, int], int]: # need to check whether the order of the steps impacts the differences between kmeans and histo.
        """
        Extract colors via histogram analysis (very fast)
        Uses numpy for optimal performance
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with colors and their frequency
        """
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        color_dict = {}
        for color, count in zip(unique_colors, counts):
            color_dict[tuple(color)] = int(count)
            
        return color_dict
    
    def extract_colors_sampling(self, image: Image.Image) -> Dict[Tuple[int, int, int], int]:
        """
        Extract colors through pixel sampling (performance optimized)
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with colors as keys and frequency as values
        """
        colors = Counter()
        width, height = image.size
        
        for y in range(0, height, self.sample_factor):
            for x in range(0, width, self.sample_factor):
                color = image.getpixel((x, y))
                colors[color] += 1
                
        return dict(colors)
    
    def extract_colors_kmeans(self, image: Image.Image, k: int = 8) -> Dict[Tuple[int, int, int], int]:
        """
        Extract colors using K-means clustering for better color grouping
        
        Args:
            image: PIL Image object
            k: Number of color clusters
            
        Returns:
            Dictionary with representative colors and their weights
        """
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        color_dict = {}
        for label, count in zip(unique_labels, counts):
            color = tuple(max(0, min(255, int(c))) for c in centers[label])
            color_dict[color] = int(count)
            
        return color_dict
    
    def reduce_colors(self, colors: Dict, reduction_factor: int = 32) -> Dict:
        """
        Reduce color palette by grouping similar colors
        
        Args:
            colors: Dictionary with colors and frequencies
            reduction_factor: Factor for color reduction (32 = 8 levels per channel)
            
        Returns:
            Dictionary with reduced colors
        """
        reduced_colors = Counter()
        
        for color, count in colors.items():
            reduced_color = tuple(
                (c // reduction_factor) * reduction_factor 
                for c in color
            )
            reduced_colors[reduced_color] += count
            
        return dict(reduced_colors)
    
    def filter_colors(self, colors: Dict, min_brightness: int = 10, 
                     max_brightness: int = 245) -> Dict:
        """
        Filter out colors that are too dark or too bright
        
        Args:
            colors: Dictionary with colors and frequencies
            min_brightness: Minimum average RGB value
            max_brightness: Maximum average RGB value
            
        Returns:
            Filtered color dictionary
        """
        filtered_colors = {}
        
        for color, count in colors.items():
            clamped_color = tuple(max(0, min(255, int(c))) for c in color)
            avg_brightness = sum(clamped_color) / 3
            if min_brightness <= avg_brightness <= max_brightness:
                filtered_colors[clamped_color] = count
        
        return filtered_colors
    
    def get_dominant_colors(self, colors: Dict, top_n: int = 10) -> List[Tuple]:
        """
        Find the most dominant colors
        
        Args:
            colors: Dictionary with colors and frequencies  
            top_n: Number of top colors to return
            
        Returns:
            List of tuples (color, frequency, percentage)
        """
        total_pixels = sum(colors.values())
        sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
        
        dominant_colors = []
        for color, count in sorted_colors[:top_n]:
            percentage = (count / total_pixels) * 100
            dominant_colors.append((color, count, percentage))
            
        return dominant_colors
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex string to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hsl(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to HSL"""
        r, g, b = [x/255.0 for x in rgb]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        l = (max_val + min_val) / 2
        
        if diff == 0:
            h = s = 0
        else:
            s = diff / (2 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
            
            if max_val == r:
                h = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / diff + 2
            else:
                h = (r - g) / diff + 4
            h /= 6
        
        return (int(h*360), int(s*100), int(l*100))
    
    @staticmethod
    def get_color_name(rgb: Tuple[int, int, int]) -> str:
        """
        Get approximate color name based on RGB values
        Simple color naming based on HSL values
        """
        h, s, l = ColorAnalyzer.rgb_to_hsl(rgb)
        
        if l < 20:
            return "Very Dark"
        elif l > 80:
            return "Very Light" 
        elif s < 20:
            return "Gray"
        else:
            if h < 15 or h >= 345:
                return "Red"
            elif h < 45:
                return "Orange"
            elif h < 75:
                return "Yellow"
            elif h < 150:
                return "Green"
            elif h < 210:
                return "Cyan"
            elif h < 270:
                return "Blue"
            elif h < 315:
                return "Purple"
            else:
                return "Pink"
    
    def analyze_image(self, image_path: Union[str, Path], 
                     top_colors: int = 10, 
                     method: str = 'histogram', 
                     reduce_palette: bool = True,
                     filter_extremes: bool = False, 
                     resize_large: bool = True) -> Dict:
        """
        Main function for image analysis
        
        Args:
            image_path: Path to image
            top_colors: Number of top colors to return
            method: Analysis method ('histogram', 'sampling', 'kmeans')
            reduce_palette: Group similar colors
            filter_extremes: Filter very dark/bright colors
            resize_large: Resize large images for performance
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        # Load image
        image, had_alpha = self.load_image(image_path)
        original_size = image.size
        
        # Resize if needed
        was_resized = False
        if resize_large:
            image, was_resized = self.resize_image_if_large(image)
        
        # Extract colors based on method
        if method == 'histogram':
            colors = self.extract_colors_histogram(image)
        elif method == 'sampling':
            colors = self.extract_colors_sampling(image)
        elif method == 'kmeans':
            colors = self.extract_colors_kmeans(image, k=top_colors * 2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        unique_colors_count = len(colors)
        
        # Apply filters
        if filter_extremes:
            colors = self.filter_colors(colors)
        
        # Reduce palette if needed
        if reduce_palette and len(colors) > 1000 and method != 'kmeans':
            colors = self.reduce_colors(colors)
        
        # Find dominant colors
        dominant_colors = self.get_dominant_colors(colors, top_colors)
        
        # Prepare result
        result = {
            'filename': image_path.name,
            'filepath': str(image_path),
            'original_size': original_size,
            'was_resized': was_resized,
            'had_alpha_channel': had_alpha,
            'analysis_method': method,
            'unique_colors': unique_colors_count,
            'filtered_colors': len(colors),
            'dominant_colors': dominant_colors,
            'processing_time': time.time() - start_time,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result


class AnalysisResult:
    """
    Container for analysis results with helper methods
    """
    
    def __init__(self, data: Dict):
        self.data = data
        self.filename = data['filename']
        self.filepath = data['filepath']
        self.original_size = data['original_size']
        self.analysis_method = data['analysis_method']
        self.dominant_colors = data['dominant_colors']
        self.processing_time = data['processing_time']
        self.timestamp = data['timestamp']
        self.had_alpha_channel = data.get('had_alpha_channel', False)
    
    def get_color_data(self, index: int) -> Dict:
        """Get detailed data for a specific color"""
        if 0 <= index < len(self.dominant_colors):
            color, count, percentage = self.dominant_colors[index]
            return {
                'rank': index + 1,
                'rgb': color,
                'hex': ColorAnalyzer.rgb_to_hex(color),
                'hsl': ColorAnalyzer.rgb_to_hsl(color),
                'color_name': ColorAnalyzer.get_color_name(color),
                'count': count,
                'percentage': percentage
            }
        return None
    
    def get_all_colors_data(self) -> List[Dict]:
        """Get detailed data for all dominant colors"""
        return [self.get_color_data(i) for i in range(len(self.dominant_colors))]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'timestamp': self.timestamp,
            'image_info': {
                'filename': self.filename,
                'filepath': self.filepath,
                'original_size': self.original_size,
                'analysis_method': self.analysis_method,
                'processing_time': round(self.processing_time, 2),
                'had_alpha_channel': self.had_alpha_channel,
            },
            'colors': self.get_all_colors_data()
        }
