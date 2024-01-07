import pickle
import numpy as np


import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

def visualize_points_in_on_img(image_path, texts, visualization_points):
    # Load the image
    img = Image.open(image_path)

    # Create a plot
    plt.imshow(img)
    palette = sns.color_palette("Paired")


    legend_entries = {}



    for i in range(len(texts)):
        text = texts[i]
        #print(text)
        color = palette[i % len(palette)]
        if text not in legend_entries:
            legend_entries[text] = color

        for point in visualization_points[text]:
            pixel_x = point[0]
            pixel_y = point[1]
            plt.scatter([pixel_x], [pixel_y], c=color)

        #plt.text(pixel_x, pixel_y, text, color=color)  # Annotate the pixel
    for text, color in legend_entries.items():
        plt.scatter([], [], c=[color], label=text)  # Dummy scatter for legend

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Show the plot
    plt.show()



def vis(image_path, label_path):
    label_path = os.path.join(label_path)
    with open(label_path, 'rb') as f:
        data_2d = pickle.load(f)
    #keys = data_2d.keys()
    annotate_texts = data_2d["verb plus noun"]
    points = data_2d["points"]


    interaction_clusters = []
    for i in range(len(annotate_texts)): #Before good_interactions
        if annotate_texts[i] not in interaction_clusters:
            interaction_clusters.append(annotate_texts[i])
    visualization_points = {}

    for i in range(len(interaction_clusters)):
        #print(interaction_clusters[i])
        visualization_points[interaction_clusters[i]] = []
        for id in range(len(annotate_texts)):
            if annotate_texts[id] == interaction_clusters[i]:
                visualization_points[interaction_clusters[i]].append(points[id])


    #print(interaction_clusters)
    visualize_points_in_on_img(image_path, interaction_clusters, visualization_points)


image_path = './P03_EPIC_100_example/rgb/P03_101_frame_0000000626.jpg'
complex_label_path = './P03_EPIC_100_example/complex_EPIC_Aff/P03_101_frame_0000000626.pkl'
easy_label_path = './P03_EPIC_100_example/easy_EPIC_Aff/P03_101_frame_0000000626.pkl'



import os

vis(image_path, complex_label_path)
vis(image_path, easy_label_path)


