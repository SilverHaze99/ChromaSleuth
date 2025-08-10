import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
import cv2


class ColorAnalyzer:
    def __init__(self, sample_factor: int = 10, enable_logging: bool = False):
        """
        Initialize the ColorAnalyzer
        
        Args:
            sample_factor: Analyze every n-th pixel (performance optimization)
            enable_logging: Enable detailed logging
        """
        self.sample_factor = sample_factor
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
        
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if the file format is supported"""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image and convert it to RGB
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object in RGB mode
            
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
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode != 'RGB':
                self.logger.info(f"Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")
    
    def resize_image_if_large(self, image: Image.Image, max_size: int = 1920) -> Image.Image:
        """
        Resize image if it's larger than max_size to improve performance
        
        Args:
            image: PIL Image object
            max_size: Maximum width/height in pixels
            
        Returns:
            Resized image if necessary, otherwise original
        """
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            self.logger.info(f"Resizing image from {width}x{height} to {new_size[0]}x{new_size[1]}")
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    
    def extract_colors_sampling(self, image: Image.Image) -> Counter:
        """
        Extract colors through pixel sampling (performance optimized)
        
        Args:
            image: PIL Image object
            
        Returns:
            Counter with colors as keys and frequency as values
        """
        colors = Counter()
        width, height = image.size
        
        # Sampling: every n-th pixel
        for y in range(0, height, self.sample_factor):
            for x in range(0, width, self.sample_factor):
                color = image.getpixel((x, y))
                colors[color] += 1
                
        return colors
    
    def extract_colors_histogram(self, image: Image.Image) -> Dict[Tuple[int, int, int], int]:
        """
        Extract colors via histogram analysis (very fast)
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with colors and their frequency
        """
        # Convert to numpy array for better performance
        img_array = np.array(image)
        
        # Reshape to 1D array of RGB tuples
        pixels = img_array.reshape(-1, 3)
        
        # Create unique colors and count them
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Convert to dictionary
        color_dict = {}
        for color, count in zip(unique_colors, counts):
            color_dict[tuple(color)] = int(count)
            
        return color_dict
    
    def extract_colors_kmeans(self, image: Image.Image, k: int = 8) -> Dict[Tuple[int, int, int], int]:
        """
        Extract colors using K-means clustering for better color grouping
        
        Args:
            image: PIL Image object
            k: Number of color clusters
            
        Returns:
            Dictionary with representative colors and their weights
        """
        # Convert to numpy array
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count pixels in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create color dictionary with clamped RGB values
        color_dict = {}
        for label, count in zip(unique_labels, counts):
            # Clamp values to valid RGB range (0-255)
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
            # Reduce each RGB value
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
            # Clamp RGB values to valid range and calculate brightness
            clamped_color = tuple(max(0, min(255, int(c))) for c in color)
            avg_brightness = sum(clamped_color) / 3
            if min_brightness <= avg_brightness <= max_brightness:
                # Use the clamped color for consistency
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
        
        # Sort by frequency
        sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
        
        dominant_colors = []
        for i, (color, count) in enumerate(sorted_colors[:top_n]):
            percentage = (count / total_pixels) * 100
            dominant_colors.append((color, count, percentage))
            
        return dominant_colors
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex string"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def rgb_to_hsl(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert RGB to HSL"""
        r, g, b = [x/255.0 for x in rgb]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2
        
        if diff == 0:
            h = s = 0  # Achromatic
        else:
            # Saturation
            s = diff / (2 - max_val - min_val) if l > 0.5 else diff / (max_val + min_val)
            
            # Hue
            if max_val == r:
                h = (g - b) / diff + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / diff + 2
            else:
                h = (r - g) / diff + 4
            h /= 6
        
        return (int(h*360), int(s*100), int(l*100))
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Get approximate color name based on RGB values
        Simple color naming based on HSL values
        """
        h, s, l = self.rgb_to_hsl(rgb)
        
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
    
    def create_color_palette(self, dominant_colors: List[Tuple], 
                           output_path: Optional[str] = None,
                           show_color_names: bool = False) -> None:
        """
        Create a visual color palette
        
        Args:
            dominant_colors: List of dominant colors
            output_path: Path to save the palette (optional)
            show_color_names: Include color names in the visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Main color bars
        bar_width = 1.0
        colors_normalized = []
        
        for i, (color, count, percentage) in enumerate(dominant_colors):
            rgb_normalized = [c/255.0 for c in color]
            colors_normalized.append(rgb_normalized)
            
            # Create bars
            ax1.barh(i, percentage, color=rgb_normalized, 
                    height=bar_width, alpha=0.9, edgecolor='white', linewidth=0.5)
            
            # Labels with color info
            hex_color = self.rgb_to_hex(color)
            color_name = self.get_color_name(color) if show_color_names else ""
            label = f"{hex_color} {color_name} ({percentage:.1f}%)"
            
            # Determine text color based on brightness
            brightness = sum(int(c) for c in color) / 3
            text_color = 'white' if brightness < 128 else 'black'
            
            ax1.text(percentage/2, i, label, va='center', ha='center', 
                    fontsize=10, weight='bold', color=text_color)
        
        ax1.set_xlabel('Percentage (%)', fontsize=12)
        ax1.set_ylabel('Colors (dominant to less dominant)', fontsize=12)
        ax1.set_title('Dominant Colors in Image', fontsize=14, weight='bold')
        ax1.set_xlim(0, max([p[2] for p in dominant_colors]) * 1.1)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Color swatches below
        swatch_width = 1.0 / len(dominant_colors)
        for i, (color, count, percentage) in enumerate(dominant_colors):
            rgb_normalized = [c/255.0 for c in color]
            ax2.add_patch(plt.Rectangle((i * swatch_width, 0), swatch_width, 1, 
                                      facecolor=rgb_normalized, edgecolor='white', linewidth=1))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Color Palette', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Color palette saved: {output_path}")
        
        plt.show()
    
    def export_colors_json(self, dominant_colors: List[Tuple], 
                          output_path: str, image_info: Dict = None) -> None:
        """
        Export color analysis results to JSON format
        
        Args:
            dominant_colors: List of dominant colors
            output_path: Path to save JSON file
            image_info: Additional image information
        """
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_info": image_info or {},
            "colors": []
        }
        
        for i, (color, count, percentage) in enumerate(dominant_colors):
            color_data = {
                "rank": i + 1,
                "rgb": color,
                "hex": self.rgb_to_hex(color),
                "hsl": self.rgb_to_hsl(color),
                "color_name": self.get_color_name(color),
                "count": count,
                "percentage": round(percentage, 2)
            }
            data["colors"].append(color_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Color data exported to: {output_path}")
    
    def print_color_analysis(self, dominant_colors: List[Tuple], 
                           image_path: Union[str, Path],
                           show_color_names: bool = False) -> None:
        """Print color analysis to console"""
        print(f"\n{'='*70}")
        print(f"COLOR ANALYSIS: {Path(image_path).name}")
        print(f"{'='*70}")
        
        header = f"{'Rank':<4} {'Hex Code':<9} {'RGB':<15} {'Percentage':<10} {'Count':<10}"
        if show_color_names:
            header += " {'Color Name':<12}"
        print(header)
        print("-" * 70)
        
        for i, (color, count, percentage) in enumerate(dominant_colors, 1):
            rgb_str = f"({color[0]:3d},{color[1]:3d},{color[2]:3d})"
            hex_color = self.rgb_to_hex(color)
            
            row = f"{i:<4} {hex_color:<9} {rgb_str:<15} {percentage:7.1f}%   {count:8,d}"
            if show_color_names:
                row += f"  {self.get_color_name(color):<12}"
            print(row)
    
    def batch_analyze(self, image_paths: List[Union[str, Path]], 
                     **kwargs) -> Dict[str, List[Tuple]]:
        """
        Analyze multiple images in batch
        
        Args:
            image_paths: List of image paths
            **kwargs: Arguments passed to analyze_image
            
        Returns:
            Dictionary mapping image paths to their dominant colors
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for path in image_paths:
                if self.is_supported_format(path):
                    future = executor.submit(self.analyze_image, path, **kwargs)
                    futures[future] = str(path)
                else:
                    print(f"Skipping unsupported format: {path}")
            
            for future in futures:
                path = futures[future]
                try:
                    result = future.result()
                    results[path] = result
                    print(f"✓ Completed: {Path(path).name}")
                except Exception as e:
                    print(f"✗ Failed {Path(path).name}: {e}")
        
        return results
    
    def analyze_image(self, image_path: Union[str, Path], top_colors: int = 10, 
                     method: str = 'histogram', reduce_palette: bool = True,
                     filter_extremes: bool = False, resize_large: bool = True,
                     show_palette: bool = True, save_palette: Optional[str] = None,
                     export_json: Optional[str] = None, 
                     show_color_names: bool = False) -> List[Tuple]:
        """
        Main function for image analysis with enhanced options
        
        Args:
            image_path: Path to image
            top_colors: Number of top colors to return
            method: Analysis method ('histogram', 'sampling', 'kmeans')
            reduce_palette: Group similar colors
            filter_extremes: Filter very dark/bright colors
            resize_large: Resize large images for performance
            show_palette: Display color palette
            save_palette: Path to save palette image
            export_json: Path to export JSON data
            show_color_names: Show color names
            
        Returns:
            List of dominant colors
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        print(f"Analyzing image: {image_path.name}")
        
        # Load image
        image = self.load_image(image_path)
        original_size = image.size
        print(f"Image size: {original_size[0]}x{original_size[1]} pixels")
        
        # Resize if needed
        if resize_large:
            image = self.resize_image_if_large(image)
        
        # Extract colors based on method
        if method == 'histogram':
            print("Using histogram analysis...")
            colors = self.extract_colors_histogram(image)
        elif method == 'sampling':
            print(f"Using pixel sampling (factor: {self.sample_factor})...")
            colors = dict(self.extract_colors_sampling(image))
        elif method == 'kmeans':
            print("Using K-means clustering...")
            colors = self.extract_colors_kmeans(image, k=top_colors * 2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Found unique colors: {len(colors):,}")
        
        # Apply filters
        if filter_extremes:
            print("Filtering extreme colors...")
            colors = self.filter_colors(colors)
            print(f"Colors after filtering: {len(colors):,}")
        
        # Reduce palette if needed
        if reduce_palette and len(colors) > 1000 and method != 'kmeans':
            print("Reducing color palette...")
            colors = self.reduce_colors(colors)
            print(f"Reduced colors: {len(colors):,}")
        
        # Find dominant colors
        dominant_colors = self.get_dominant_colors(colors, top_colors)
        
        # Output results
        self.print_color_analysis(dominant_colors, image_path, show_color_names)
        
        # Create visualizations and exports
        if show_palette or save_palette:
            self.create_color_palette(dominant_colors, save_palette, show_color_names)
        
        if export_json:
            image_info = {
                "filename": image_path.name,
                "original_size": original_size,
                "analysis_method": method,
                "total_unique_colors": len(colors)
            }
            self.export_colors_json(dominant_colors, export_json, image_info)
        
        elapsed_time = time.time() - start_time
        print(f"\nAnalysis completed in {elapsed_time:.2f} seconds!")
        print(f"Found {len(dominant_colors)} dominant colors.")
        
        return dominant_colors


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Color Analysis Tool - Analyze dominant colors in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                          # Basic analysis
  %(prog)s image.png -n 5 -o palette.png     # Top 5 colors, save palette
  %(prog)s logo.jpg -m kmeans --color-names  # K-means with color names
  %(prog)s *.jpg --batch -j results.json     # Batch process with JSON export
        """
    )
    
    parser.add_argument('image_path', nargs='+', help='Path(s) to image file(s)')
    parser.add_argument('-n', '--top-colors', type=int, default=10, 
                       help='Number of top colors (default: 10)')
    parser.add_argument('-m', '--method', choices=['histogram', 'sampling', 'kmeans'],
                       default='histogram', help='Analysis method (default: histogram)')
    parser.add_argument('--sample-factor', type=int, default=10,
                       help='Sampling factor for sampling method (default: 10)')
    parser.add_argument('--no-reduce', action='store_true',
                       help='Don\'t apply color reduction')
    parser.add_argument('--filter-extremes', action='store_true',
                       help='Filter very dark/bright colors')
    parser.add_argument('--no-resize', action='store_true',
                       help='Don\'t resize large images')
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t show color palette')
    parser.add_argument('--color-names', action='store_true',
                       help='Show color names in output')
    parser.add_argument('-o', '--output', help='Save color palette as image')
    parser.add_argument('-j', '--json', help='Export results to JSON file')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images in batch mode')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Check if files exist
    image_paths = []
    for path_str in args.image_path:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                image_paths.append(path)
            elif path.is_dir():
                # Add all supported images in directory
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    image_paths.extend(path.glob(f'*{ext}'))
                    image_paths.extend(path.glob(f'*{ext.upper()}'))
        else:
            print(f"Warning: File not found: {path}")
    
    if not image_paths:
        print("Error: No valid image files found!")
        sys.exit(1)
    
    # Create analyzer
    analyzer = ColorAnalyzer(
        sample_factor=args.sample_factor,
        enable_logging=args.verbose
    )
    
    try:
        if args.batch and len(image_paths) > 1:
            print(f"Batch processing {len(image_paths)} images...")
            results = analyzer.batch_analyze(
                image_paths,
                top_colors=args.top_colors,
                method=args.method,
                reduce_palette=not args.no_reduce,
                filter_extremes=args.filter_extremes,
                resize_large=not args.no_resize,
                show_palette=False,  # Don't show individual palettes in batch mode
                show_color_names=args.color_names
            )
            
            print(f"\nBatch processing completed! Processed {len(results)} images.")
            
            if args.json:
                # Export batch results
                batch_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_images": len(results),
                    "results": {}
                }
                
                for path, colors in results.items():
                    batch_data["results"][Path(path).name] = [
                        {
                            "rank": i + 1,
                            "rgb": color[0],
                            "hex": analyzer.rgb_to_hex(color[0]),
                            "percentage": round(color[2], 2)
                        } for i, color in enumerate(colors)
                    ]
                
                with open(args.json, 'w') as f:
                    json.dump(batch_data, f, indent=2)
                print(f"Batch results exported to: {args.json}")
        
        else:
            # Single image analysis
            for image_path in image_paths:
                dominant_colors = analyzer.analyze_image(
                    image_path=image_path,
                    top_colors=args.top_colors,
                    method=args.method,
                    reduce_palette=not args.no_reduce,
                    filter_extremes=args.filter_extremes,
                    resize_large=not args.no_resize,
                    show_palette=not args.no_display,
                    save_palette=args.output,
                    export_json=args.json,
                    show_color_names=args.color_names
                )
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        print("Enhanced Color Analysis Tool")
        print("============================")
        print("Usage: python color_analysis.py <image_path> [options]")
        print("\nFor detailed help: python color_analysis.py -h")
        print("\nQuick examples:")
        print("  python color_analysis.py image.jpg")
        print("  python color_analysis.py *.png --batch -j results.json")
        print("  python color_analysis.py logo.jpg -m kmeans --color-names")
    else:
        main()
