

# def gen_vid_from_dir(delay = 5):


# import cv2
# import os

def create_video(image_folder, video_name, duration=5, frame_size=(1920, 1080)):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, 1/duration, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        frame = cv2.resize(frame, frame_size)
        for _ in range(duration):
            video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# Usage
create_video('path_to_image_directory', 'output_video.mp4')
