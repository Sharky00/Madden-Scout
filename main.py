import cv2
from get_video import get_video
from capture_manager import capture_manager

if __name__ == "__main__":
    GV = get_video()
    CM = capture_manager(cv2.VideoCapture(GV.vid_file))

    CM.output_dir()
    CM.seek_video(400)
    CM.run_analysis()

