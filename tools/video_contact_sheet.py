import argparse
import math
import os

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="output/downloads/source_video.mp4")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=420.0)
    parser.add_argument("--step", type=float, default=15.0)
    parser.add_argument("--output", default="output/contact_sheet_0_420.jpg")
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    thumbnails = []
    timestamp = args.start
    while timestamp <= args.end:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if ok:
            thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            cv2.rectangle(thumb, (0, 0), (100, 24), (0, 0, 0), -1)
            cv2.putText(
                thumb,
                f"{int(timestamp) // 60}:{int(timestamp) % 60:02d}",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            thumbnails.append(thumb)
        timestamp += args.step
    capture.release()

    columns = 4
    rows = math.ceil(len(thumbnails) / columns)
    sheet = 255 * __import__("numpy").ones(
        (rows * 180, columns * 320, 3), dtype="uint8"
    )
    for index, thumb in enumerate(thumbnails):
        row, column = divmod(index, columns)
        sheet[row * 180 : (row + 1) * 180, column * 320 : (column + 1) * 320] = thumb

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, sheet)
    print(args.output)


if __name__ == "__main__":
    main()
