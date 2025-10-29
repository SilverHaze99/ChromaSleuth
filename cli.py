import sys
import json
import time
from pathlib import Path
from typing import List
import argparse
from concurrent.futures import ThreadPoolExecutor

from main import ColorAnalyzer, AnalysisResult


class CLI:
    """Command line interface handler"""
    
    def __init__(self):
        self.analyzer = None
    
    def print_header(self, text: str, char: str = "="):
        """Print formatted header"""
        width = 70
        print(f"\n{char * width}")
        print(f"{text:^{width}}")
        print(f"{char * width}\n")
    
    def print_color_analysis(self, result: AnalysisResult, show_color_names: bool = False):
        """Print color analysis to console"""
        self.print_header(f"COLOR ANALYSIS: {result.filename}")
        
        print(f"Image Size:    {result.original_size[0]} x {result.original_size[1]} px")

        if result.had_alpha_channel:
            print(f"Tranyparency: Yes (Flattened for analysis)")
            
        print(f"Method:        {result.analysis_method.capitalize()}")
        print(f"Processing:    {result.processing_time:.2f} seconds")
        print(f"Total Colors:  {result.data['unique_colors']:,}")
        
        if result.data.get('filtered_colors'):
            print(f"After Filter:  {result.data['filtered_colors']:,}")
        
        print(f"\n{'Rank':<6}{'Hex':<10}{'RGB':<18}{'Percentage':<12}{'Count':<12}", end="")
        if show_color_names:
            print(f"{'Color Name':<15}", end="")
        print()
        print("-" * 70)
        
        for i in range(len(result.dominant_colors)):
            color_data = result.get_color_data(i)
            rgb = color_data['rgb']
            rgb_str = f"({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d})"
            
            print(f"{color_data['rank']:<6}{color_data['hex']:<10}{rgb_str:<18}"
                  f"{color_data['percentage']:>6.2f}%     {color_data['count']:>8,}  ", end="")
            
            if show_color_names:
                print(f"  {color_data['color_name']:<15}", end="")
            print()
        
        print()
    
    def export_json(self, result: AnalysisResult, output_path: str):
        """Export results to JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Results exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting JSON: {e}")
    
    def export_palette(self, result: AnalysisResult, output_path: str):
        """Export color palette as image"""
        from PIL import Image, ImageDraw, ImageFont
        
        try:
            width = 800
            height = len(result.dominant_colors) * 80 + 100
            
            img = Image.new('RGB', (width, height), color=(18, 18, 18))
            draw = ImageDraw.Draw(img)
            
            try:
                font_large = ImageFont.truetype("arial.ttf", 16)
                font_small = ImageFont.truetype("arial.ttf", 12)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            y_offset = 20
            
            for i in range(len(result.dominant_colors)):
                color_data = result.get_color_data(i)
                rgb = color_data['rgb']
                
                draw.rectangle([20, y_offset, 120, y_offset + 60], fill=rgb)
                
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
            print(f"Color palette saved: {output_path}")
        except Exception as e:
            print(f"Error exporting palette: {e}")
    
    def analyze_single(self, image_path: str, args):
        """Analyze a single image"""
        print(f"Analyzing: {Path(image_path).name}")
        
        try:
            result_data = self.analyzer.analyze_image(
                image_path=image_path,
                top_colors=args.top_colors,
                method=args.method,
                reduce_palette=not args.no_reduce,
                filter_extremes=args.filter_extremes,
                resize_large=not args.no_resize
            )
            
            result = AnalysisResult(result_data)
            
            self.print_color_analysis(result, args.color_names)
            
            if args.json:
                json_path = args.json
                if args.batch:
                    json_path = str(Path(image_path).with_suffix('.json'))
                self.export_json(result, json_path)
            
            if args.output:
                output_path = args.output
                if args.batch:
                    output_path = str(Path(image_path).stem + '_palette.png')
                self.export_palette(result, output_path)
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {Path(image_path).name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def analyze_batch(self, image_paths: List[str], args):
        """Analyze multiple images"""
        self.print_header(f"BATCH ANALYSIS: {len(image_paths)} images")
        
        results = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.analyze_single, path, args): path 
                      for path in image_paths}
            
            for future in futures:
                path = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[path] = result
                        print(f"Completed: {Path(path).name}")
                except Exception as e:
                    print(f"Failed: {Path(path).name} - {e}")
        
        total_time = time.time() - start_time
        
        self.print_header("BATCH SUMMARY")
        print(f"Total images:     {len(image_paths)}")
        print(f"Successful:       {len(results)}")
        print(f"Failed:           {len(image_paths) - len(results)}")
        print(f"Total time:       {total_time:.2f} seconds")
        print(f"Average per image: {total_time/len(image_paths):.2f} seconds\n")
        
        if args.json and len(results) > 1:
            self.export_batch_json(results, args.json)
    
    def export_batch_json(self, results: dict, output_path: str):
        """Export batch results to JSON"""
        batch_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images": len(results),
            "results": {}
        }
        
        for path, result in results.items():
            batch_data["results"][Path(path).name] = result.to_dict()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            print(f"Batch results exported to: {output_path}")
        except Exception as e:
            print(f"Error exporting batch JSON: {e}")
    
    def run(self, args):
        """Main CLI execution"""
        self.analyzer = ColorAnalyzer(
            sample_factor=args.sample_factor,
            enable_logging=args.verbose
        )
        
        image_paths = []
        for path_str in args.image_path:
            path = Path(path_str)
            if path.exists():
                if path.is_file() and ColorAnalyzer.is_supported_format(path):
                    image_paths.append(str(path))
                elif path.is_dir():
                    for ext in ColorAnalyzer.SUPPORTED_FORMATS:
                        image_paths.extend([str(p) for p in path.glob(f'*{ext}')])
            else:
                print(f"Warning: File not found: {path}")
        
        if not image_paths:
            print("Error: No valid image files found")
            sys.exit(1)
        
        if args.batch and len(image_paths) > 1:
            self.analyze_batch(image_paths, args)
        else:
            for image_path in image_paths:
                self.analyze_single(image_path, args)


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='ChromaSleuth - Analyze dominant colors in images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  chromasleuth image.jpg
  chromasleuth image.png -n 5 -o palette.png
  chromasleuth logo.jpg -m kmeans --color-names
  chromasleuth *.jpg --batch -j results.json

Methods:
  histogram - Fast histogram analysis (default)
  sampling  - Pixel sampling for large images
  kmeans    - K-means clustering for best quality
        """
    )
    
    parser.add_argument('image_path', nargs='+', 
                       help='Path(s) to image file(s) or directory')
    
    parser.add_argument('-n', '--top-colors', type=int, default=10, 
                       metavar='N',
                       help='Number of top colors to extract (default: 10)')
    
    parser.add_argument('-m', '--method', 
                       choices=['histogram', 'sampling', 'kmeans'],
                       default='histogram',
                       help='Analysis method (default: histogram)')
    
    parser.add_argument('--sample-factor', type=int, default=10,
                       metavar='N',
                       help='Sampling factor for sampling method (default: 10)')
    
    parser.add_argument('--no-reduce', action='store_true',
                       help='Disable color reduction for similar colors')
    
    parser.add_argument('--filter-extremes', action='store_true',
                       help='Filter very dark and very bright colors')
    
    parser.add_argument('--no-resize', action='store_true',
                       help='Disable automatic resizing of large images')
    
    parser.add_argument('--color-names', action='store_true',
                       help='Show approximate color names in output')
    
    parser.add_argument('-o', '--output', metavar='FILE',
                       help='Save color palette as PNG image')
    
    parser.add_argument('-j', '--json', metavar='FILE',
                       help='Export analysis results to JSON file')
    
    parser.add_argument('--batch', action='store_true',
                       help='Enable batch processing mode for multiple images')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging output')
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        print("ChromaSleuth - Color Analysis Tool")
        print("=" * 50)
        print("\nUsage: python cli.py <image_path> [options]")
        print("\nFor detailed help: python cli.py -h")
        print("\nQuick Examples:")
        print("  python cli.py image.jpg")
        print("  python cli.py *.png --batch -j results.json")
        print("  python cli.py logo.jpg -m kmeans --color-names\n")
        sys.exit(0)
    
    args = parser.parse_args()
    
    cli = CLI()
    
    try:
        cli.run(args)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()