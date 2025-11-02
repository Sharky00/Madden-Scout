import cv2, time
import process_video
from pathlib import Path

# thresholds
DETECT_THR   = 0.60   # consider panel present if >= this
SAVE_THR     = 0.72   # only save if >= this (crisper)
DIFF_THR     = 0.012  # <= ~1.2% avg gray change between frames
SHARP_THR    = 150.0  # Laplacian variance (tune 120..250)
N_STABLE     = 3      # require N consecutive stable frames
COOLDOWN_FR  = 30     # avoid multiple saves while panel stays up
TARGET_W, TARGET_H = 1280, 720

class capture_manager:
    def __init__(self, cap):    
        self.PV = process_video
        self.cap = cap
        self.prev_roi_gray = None
        self.stable_count  = 0
        self.cooldown      = 0

    def run_analysis(self):
        while self.cap.isOpened():
            ok, img = self.cap.read()
            if not ok: break

            disp = cv2.resize(img, (TARGET_W, TARGET_H))
            t_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

            # decide to save based on yard_line sameness vs previous frame


            score, roi_gray, roi_bgr = self.PV.match_prev_play(disp, self.tpl)

            if score >= DETECT_THR:
                is_stable, d = self.PV.panel_stable(roi_gray, self.prev_roi_gray, diff_thr=DIFF_THR)
                is_sharp,  s = self.PV.sharp_enough(roi_gray, sharp_thr=SHARP_THR)

                if score >= SAVE_THR and is_stable and is_sharp and self.cooldown == 0:
                    self.stable_count += 1
                else:
                    self.stable_count = 0

                if self.stable_count >= N_STABLE and self.cooldown == 0:
                    frame_dir = self.out_root / f"t{t_ms:012d}_f{self.frame_idx:09d}"
                    yard_line = self.PV.yardline_similar_to_prev(disp,"yard_line", .95, .4)
                    down_distance = self.PV.yardline_similar_to_prev(disp,"down_distance", .95, .4)

                    if yard_line or down_distance :
                        self.PV.save_hud_regions_to_dir(disp, frame_dir)
                    self.cooldown = COOLDOWN_FR
                    self.stable_count = 0  # reset so we don't double-save

            else:
                self.stable_count = 0
                self.prev_roi_gray = None

            # update previous & cooldown
            self.prev_roi_gray = roi_gray
            if self.cooldown > 0:
                self.cooldown -= 1

            cv2.imshow("canOut", disp)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
            self.frame_idx += 1

        self.cap.release()
        cv2.destroyAllWindows()

    def output_dir(self):
        if not self.cap.isOpened():
            print("Error Opening video"); raise SystemExit

        self.tpl = cv2.imread("templates/prev_play_title.jpg")
        if self.tpl is None:
            raise FileNotFoundError("templates/prev_play_title.png not found")

        self.out_root = Path("frames_out") / time.strftime("%Y%m%d_%H%M%S")
        self.out_root.mkdir(parents=True, exist_ok=True)

    def seek_video(self, seek):
        seek_time = seek * 50

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_time)
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def retrieve_cap(self):
        return self.cap
    
    @property
    def retrieve_tpl(self):
        return self.tpl
    
    @property
    def retrieve_out_root(self):
        return self.out_root