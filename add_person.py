"""
Script to add a new person by capturing images from webcam and saving to Data folder.
"""

import cv2
import os

def add_new_person():
    """
    Capture images for a new person and save to Data/person_name/ folder.
    """
    # Input person name
    person_name = input("Enter the name of the new person: ").strip()
    if not person_name:
        print("Name cannot be empty.")
        return

    # Create directory for the person
    data_dir = './Data'
    person_dir = os.path.join(data_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print(f"Capturing images for {person_name}. Press SPACE to capture, 'q' to quit.")
    print("Try to capture 10-20 images from different angles.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Display instructions on frame
        cv2.putText(frame, f"Capturing for: {person_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Images captured: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to capture, 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Add New Person', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space to capture
            count += 1
            img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Captured image {count}: {img_path}")
        elif key == ord('q'):  # q to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Finished capturing. {count} images saved for {person_name}.")

if __name__ == "__main__":
    add_new_person()