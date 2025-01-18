import cv2
import mediapipe as mp
import os

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Specify the image path
image_path = r"c:/Users/B.UDAY KIRAN/OneDrive/Desktop/AICTE INTERNSHIP/Human Pose Estimation using Machine Learning/OIP.jpeg"

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Error: The file at {image_path} does not exist.")
    exit()

# Load the image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load the image. Check the file format.")
    exit()

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose estimation
results = pose.process(image_rgb)

if results.pose_landmarks:
    print("Pose landmarks detected!")
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z}, visibility: {landmark.visibility})")

    # Draw landmarks
    h, w, c = image.shape
    for landmark in results.pose_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)  # Green circles

    # Save and display the annotated image
    annotated_image_path = "annotated_image.jpg"
    cv2.imwrite(annotated_image_path, image)
    print(f"Annotated image saved to {annotated_image_path}")

    cv2.imshow("Pose Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No pose landmarks detected.")

# Release resources
pose.close()
