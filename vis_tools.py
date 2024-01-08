
import cv2
import os

def remove_filename_whitespace(directory):
  for filename in os.listdir(directory):
    new_filename = filename.replace(' ', '_')
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
    print(f"Renamed '{filename}' to '{new_filename}'")

def create_video(image_folder, video_name, duration=3):
    _files = os.listdir(image_folder)
    _files.sort()
    images = [img for img in _files if img.endswith(".jpg") or img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer with XVID codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video = cv2.VideoWriter(video_name, fourcc, 1.0, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        for _ in range(int(duration * 1)):  # Increase the frame rate slightly
            video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# Usage
if __name__ == '__main__':

  create_video('data/heatmap_jan8/gmmeasy',  'data/heatmap_jan8/gmmeasy_vid.mp4')
  create_video('data/heatmap_jan8/points_easy',  'data/heatmap_jan8/points_easy_vid.mp4')
  create_video('data/heatmap_jan8/RGB',  'data/heatmap_jan8/RGB_vid.mp4')

