import cv2

VIDEO_PATH = r"match.mp4"
PREVIEW_WIDTH = 1280  # adjust if needed

cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Could not read first frame")

h, w = frame.shape[:2]
scale = PREVIEW_WIDTH / w
preview = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)))

roi_preview = cv2.selectROI("Select CLOCK ROI (resized)", preview, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, rw, rh = roi_preview
# Scale ROI back to original frame coordinates
roi_original = (int(x / scale), int(y / scale), int(rw / scale), int(rh / scale))

print("Original ROI =", roi_original)
