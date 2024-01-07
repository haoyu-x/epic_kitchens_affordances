import pickle
import numpy as np


import matplotlib.pyplot as plt
from PIL import Image
def visualize_points(image_path, text, visualization_points):
    # Load the image
    img = Image.open(image_path)

    # Create a plot
    plt.imshow(img)
    for point in visualization_points[text]:
        pixel_x = point[0]
        pixel_y = point[1]
        plt.scatter([pixel_x], [pixel_y], c='red')  # Mark the pixel with a red dot
    plt.text(pixel_x, pixel_y, text, color='white')  # Annotate the pixel

    # Show the plot
    plt.show()



image_path = './P03_EPIC_100_example/rgb/P03_101_frame_0000000626.jpg'
label_path = './P03_EPIC_100_example/complex_EPIC_Aff/P03_101_frame_0000000626.pkl'

import os
label_2d_path = os.path.join(label_path)
with open(label_2d_path, 'rb') as f:
    data_2d = pickle.load(f)

keys = data_2d.keys()
#print("Keys:", keys)

annotate_texts = data_2d["verb plus noun"]
points = data_2d["points"]

interaction_clusters = []

for i in range(len(annotate_texts)): #Before good_interactions
    if annotate_texts[i] not in interaction_clusters:
        interaction_clusters.append(annotate_texts[i])


print(interaction_clusters)


visualization_points = {}

for i in range(len(interaction_clusters)):
    print(interaction_clusters[i])
    visualization_points[interaction_clusters[i]] = []
    for id in range(len(annotate_texts)):
        if annotate_texts[id] == interaction_clusters[i]:
            visualization_points[interaction_clusters[i]].append(points[id])

print(visualization_points)
for text in interaction_clusters:
    visualize_points(image_path, text, visualization_points)



