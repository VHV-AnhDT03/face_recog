import cv2

def open_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Unable to active camera")
        return

    print("Camera actived. press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not get frame")
            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    open_camera(0)
