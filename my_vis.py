import pickle
import numpy as np


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
import os
from PIL import Image
from scipy.stats import multivariate_normal
import time
import cv2
def visualize_and_annotate(image_path, pixel_x, pixel_y, annotation_text):
    # Load the image
    img = Image.open(image_path)

    # Create a plot
    plt.imshow(img)
    plt.scatter([pixel_x], [pixel_y], c='red')  # Mark the pixel with a red dot
    plt.text(pixel_x, pixel_y, annotation_text, color='white')  # Annotate the pixel

    # Show the plot
    plt.show()

# Example usage



class Ego_Metric_training_dataset():
    def __init__(self, Ego_Metric_dataset_path=None):
        #self.main_dir = Ego_Metric_dataset_path
        #self.samples_txt = os.path.join(self.main_dir, 'samples.txt')
        self.img_dir = 'selected_plus_guided_rgb'
        self.label_2d = '2d_output_labels'
        self.label_3d = 'aff_on_3d'
        #self.valid_verbs = ['take', 'remove', 'put', 'insert', 'throw', 'wash', 'dry', 'open', 'turn-on', 
        #                    'close', 'turn-off', 'mix', 'fill', 'add', 'cut', 'peel', 'empty', 
        #                    'shake', 'squeeze', 'press', 'cook', 'move', 'adjust', 'eat', 
        #                    'drink', 'apply', 'sprinkle', 'fold', 'sort', 'clean', 'slice', 'pick']
        self.height = 480
        self.width = 854
        self.size = 500

        self.pos = self.get_pos_for_gaussian()
        self.gaussian = self.get_gaussian()
        #self.colormap_interactions = colormap_interactions
        self.interaction_clusters = []


    def __len__(self):
        return len(self.samples)  
    
    def get_pos_for_gaussian(self):
        x, y = np.mgrid[0:self.width:1, 0:self.height:1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y 
        return pos
    
    def get_gaussian(self):
        x, y = np.mgrid[0:self.size:1, 0:self.size:1]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        gaussian = multivariate_normal(mean=[self.size//2, self.size//2], cov=np.eye(2)*1000)
        return gaussian.pdf(pos)

    def get_masks_from_pickle(self, data):
        #Cluster the interactions
        #interaction_clusters = []
        interaction_coordinates = {}
        verbs_data = data['verbs']
        texts_data = data["verb plus noun"]
        points_data = data['points']

        for i in range(len(texts_data)): #Before good_interactions
            if texts_data[i] not in self.interaction_clusters:
                self.interaction_clusters.append(texts_data[i])
        for i in range(len(self.interaction_clusters)):
            interaction_coordinates[self.interaction_clusters[i]] = []
        for i in range(len(texts_data)):
            interaction_coordinates[texts_data[i]].append(points_data[i])

        print("interaction_clusters", self.interaction_clusters)

        #Draw the hotspots of the clusters
        c = 0
        masks = np.zeros((len(self.interaction_clusters), self.height, self.width))
        for text in self.interaction_clusters: 
            if text in interaction_coordinates.keys():  
                print(text)

                prob_sum = np.zeros((self.width, self.height))
                for j in range(len(interaction_coordinates[text])):
                    point = interaction_coordinates[text][j][0:2].astype(int)
                    print(point)
                    prob = np.zeros((self.width, self.height))

                    if (self.width - point[0]) > self.size // 2:
                        gauss_right = self.size
                        prob_right = point[0] + self.size // 2
                    else:
                        gauss_right = self.width - point[0] + self.size // 2
                        prob_right = self.width             


                    if point[0] > self.size // 2:
                        gauss_left = 0
                        prob_left = point[0] - self.size // 2
                    else:
                        gauss_left = self.size // 2 - point[0]
                        prob_left = 0       
                    
                    
                    if (self.height - point[1]) > self.size // 2:
                        gauss_bottom = self.size
                        prob_bottom = point[1] + self.size // 2 
                    else:
                        gauss_bottom = self.height - point[1] + self.size // 2
                        prob_bottom = self.height
                    
                    
                    if point[1] > self.size // 2:
                        gauss_top = 0
                        prob_top = point[1] - self.size // 2
                    else:
                        gauss_top = self.size // 2 - point[1]
                        prob_top = 0
                    
                    
                    
                    prob[int(prob_left):int(prob_right),int(prob_top):int(prob_bottom)] = self.gaussian[int(gauss_left):int(gauss_right),int(gauss_top):int(gauss_bottom)]
                    prob_sum += prob

                prob_sum = (prob_sum / np.max(prob_sum)).T
                prob_sum[prob_sum < 0.25] = 0 #If prob_sum < 0.5, set it to 0
                prob_sum[prob_sum >= 0.25] = 1 #If prob_sum >= 0.5, set it to 1
                masks[c, :, :] = prob_sum
                print("prob", prob_sum)

            c += 1
        return masks

    def visualize(self, img, masks, selected_text):
        img_copy = img.copy()
        #print(img_copy.shape)
        selected_verb_idx = self.interaction_clusters.index(selected_text)
        selected_mask = masks[selected_verb_idx, :, :]
        #cv2.imshow('mask', selected_mask)
        selected_mask_2 = selected_mask[:, :, np.newaxis]
        #cv2.imshow('selected_mask_2', selected_mask_2)
        #print(selected_mask_2.shape)
        #selected_mask_2 = selected_mask
        #color = np.array(self.colormap_interactions[selected_verb]).reshape(1, 3)
        color = np.array([[255], [255], [255]]).reshape(1, 3)
        prob_paint = (selected_mask_2 @ color).astype(np.uint8)
        img_copy = cv2.addWeighted(img_copy, 1.0, prob_paint, 1.0, 0)
        #cv2.imwrite(os.path.join('/home/lmur/Desktop/EGO_METRIC_Dataset_v3/Kitchens/P04_EPIC_55/show/img.png'), img_copy)
        cv2.imshow('Image', img_copy)

        # Wait for a key press and then close the displayed window
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def __getitem__(self, idx):
        kitchen, sample_id = self.samples[idx].split('/')
        #Load the image
        img_path = os.path.join(self.main_dir, kitchen, self.img_dir, sample_id + '.jpg')
        img = cv2.imread(img_path)
        #Load the labels
        label_2d_path = os.path.join(self.main_dir, kitchen, self.label_2d, sample_id + '.pkl')
        with open(label_2d_path, 'rb') as f:
            data_2d = pickle.load(f)
        masks = self.get_masks_from_pickle(data_2d)
        return img, masks

image_path = '/home/haoyux/epic_kitchens_affordances/P03_EPIC_100_example/rgb/P03_101_frame_0000000626.jpg'
label_path = '/home/haoyux/epic_kitchens_affordances/P03_EPIC_100_example/complex_EPIC_Aff/P03_101_frame_0000000626.pkl'


import os
label_2d_path = os.path.join(label_path)
with open(label_2d_path, 'rb') as f:
    data_2d = pickle.load(f)





dataset = Ego_Metric_training_dataset("")
masks = dataset.get_masks_from_pickle(data_2d)


#Load the image
img_path = os.path.join(image_path)
img = cv2.imread(img_path)

annotate_texts = data_2d["verb plus noun"]
interaction_clusters = []

for i in range(len(annotate_texts)): #Before good_interactions
    if annotate_texts[i] not in interaction_clusters:
        interaction_clusters.append(annotate_texts[i])
for text in interaction_clusters:
    print(text)
    dataset.visualize(img, masks, text)
