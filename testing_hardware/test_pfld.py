import cv2
from src.dms.modules.pfld import PFLDDetector

def csi_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, "
        "framerate=30/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )

cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)

pfld = PFLDDetector("models/pfld_106_lite.onnx")

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    result = pfld.process(frame)

    if result["EAR"] is not None:
        cv2.putText(
            frame,
            f"EAR: {result['EAR']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {result['MAR']}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("PFLD Test", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
