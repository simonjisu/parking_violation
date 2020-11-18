import argparse
from pathlib import Path
from realsense import D455

def main():
    parser = argparse.ArgumentParser(description="Record color and depth frame, pointcloud")
    parser.add_argument("-wid", "--width", type=int, default=640,
        help="Width")
    parser.add_argument("-hei", "--height", type=int, default=480,
        help="Height")
    parser.add_argument("-fps", "--framerate", type=int, default=30,
        help="FPS")
    parser.add_argument("-rt", "--record_time", type=int, default=6*10,
        help="Record Time(Sec)")
    parser.add_argument("-svp", "--sv_path", type=str, default="./saved",
        help="Save path")
    parser.add_argument("-md", "--max_dist", type=float, default=2.5,
        help="Max distance")
    parser.add_argument("-svimg", "--saveimg", action="store_true",
        help="Save Img")
    parser.add_argument("-svpc", "--savepc", action="store_true",
        help="Save Point Cloud")
    parser.add_argument("-svbag", "--savebag", action="store_true",
        help="Save the video to bag file")

    args = parser.parse_args()
    sv_path = Path().absolute() / args.sv_path
    if not sv_path.exists():
        sv_path.mkdir()
    camera = D455(
        width=args.width,
        height=args.height,
        framerate=args.framerate,
        max_dist=args.max_dist,
        sv_path=sv_path,
        record_time=args.record_time,
        saveimg=args.saveimg,
        savepc=args.savepc,
        savebag=args.savebag
    )
    camera.run_app()

if __name__ == "__main__":
    main()
