import argparse
import sys
import os
import csv

def main():
    parser = argparse.ArgumentParser(description="Fire Detection Pipeline: Baseline vs Upgraded")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--mode", choices=["baseline", "upgraded"], required=True, 
                        help="Choose the pipeline mode: 'baseline' or 'upgraded'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
        sys.exit(1)
        
    results = None
    if args.mode == "baseline":
        from baseline import run_baseline
        results = run_baseline(args.video_path)
    elif args.mode == "upgraded":
        from upgraded import run_upgraded
        results = run_upgraded(args.video_path)

    if results:
        avg_fps = sum(r["fps"] for r in results) / len(results)
        avg_contours = sum(r["contour_count"] for r in results) / len(results)
        
        print("\n" + "="*40)
        print(f"| Metrics Summary ({args.mode.upper()}) ")
        print("="*40)
        print(f"| Average FPS:          {avg_fps:.2f}")
        print(f"| Avg Contours/Frame:   {avg_contours:.2f}")
        print("="*40 + "\n")
        
        csv_file = "benchmark_results.csv"
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "fps", "contour_count", "spread_metric"])
            writer.writeheader()
            for i, r in enumerate(results):
                writer.writerow({
                    "frame": i + 1,
                    "fps": round(r["fps"], 2),
                    "contour_count": r["contour_count"],
                    "spread_metric": round(r["spread_metric"], 4)
                })
        print(f"Exported frame-by-frame data to {os.path.abspath(csv_file)}")

if __name__ == "__main__":
    main()
