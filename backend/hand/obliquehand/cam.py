import cv2

print("\n===================================")
print("SEARCHING FOR AVAILABLE CAMERAS")
print("Press Q to close camera")
print("===================================\n")

camera_found = False

# Try camera indexes 0 to 5
for i in range(6):

    print(f"Trying camera index: {i}")

    cap = cv2.VideoCapture(
        i,
        cv2.CAP_DSHOW
    )

    # Check if camera opened
    if cap.isOpened():

        print(f"✅ Camera found at index {i}")

        camera_found = True

        while True:

            ret, frame = cap.read()

            if not ret:
                print("❌ Failed to read frame")
                break

            if frame is None:
                continue

            # Show current camera index
            cv2.putText(
                frame,
                f"Camera Index: {i}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to close",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            cv2.imshow(
                f"Camera Test - Index {i}",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # Quit current camera
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:

        print(f"❌ Camera index {i} not working")

# No camera found
if not camera_found:

    print("\n❌ No camera detected")
    print("Check:")
    print("1. DroidCam app is open on phone")
    print("2. DroidCam Client is running on PC")
    print("3. Phone and PC connected")
    print("4. Camera permissions enabled")

print("\nProgram finished")