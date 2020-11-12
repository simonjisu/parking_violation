from realsense import D455
from pathlib import Path

def main():
    width, height, framerate =(640, 480, 30) 
    max_dist = 2.5
    sv_path = Path().absolute().parent / "saved"
    if not sv_path.exists():
        sv_path.mkdir()
    cm = D455(width, height, framerate, max_dist, sv_path)
    cm.run_app()

if __name__ == "__main__":
    main()
