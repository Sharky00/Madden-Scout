import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import video_intake as vi

REGIONS = {
    "defense_header": vi.Region(0.790, 0.245, 0.155, 0.035),
    "defense_formation": vi.Region(0.735, 0.277, 0.205, 0.035),
    "defense_personnel": vi.Region(0.748, 0.302, 0.170, 0.035),
    "offense_formation": vi.Region(0.055, 0.435, 0.220, 0.045),
    "offense_formation_tight": vi.Region(0.058, 0.440, 0.180, 0.030),
    "offense_formation_name": vi.Region(0.105, 0.440, 0.135, 0.030),
}


def main() -> None:
    vi.configure_tesseract(None)
    for frame_path in sys.argv[1:]:
        frame = cv2.imread(frame_path)
        print(frame_path)
        for name, region in REGIONS.items():
            crop = vi.crop_region(frame, region)
            text = vi.ocr_text_from_crop(crop, None, 7)
            print(f"  {name:20s} {text!r}")


if __name__ == "__main__":
    main()
