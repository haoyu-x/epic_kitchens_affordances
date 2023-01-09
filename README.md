# Epic kitchens affordances

## Dataset creation: automatic annotations

The EPIC-Affordance dataset is a new dataset build on the Epic Kitchens 100 and Epic Kitchens VISOR, containing **automatic annotations** generated by the intersection of both datasets. We provide **50,242** images with **123,233** different annotation masks.
You can download all the data in this link:
-Images: we already provide the images extracted from the videos of EPIC-100 Kitchens. This avoids download the approximate 700 GB of that dense dataset.
-Annotations in 3D: in a pickle format, we provide a dictionary with the Colmap data (camera pose, camera intrinsics and keypoints), the distribution of the interacting objects, the annotation of the interaction and the distribution of the neutral objects. We encourage to the research community to use this data to develop new tasks like goal path planning.
-Affordance annotations in the 2D: we already run the project_from_3D_to_2D.py for all the sequences in order to provide a pickle dictionary with the location of the interaction points for the 32 different afforded-actions.

### 1. Detect the spatial localization of the interaction

On one hand, we use the narration annotations of the Epic Kitchens 100 to obtain the semantics of the interaction (e.g "cut onion"). Then, we use the masks provided by EPIC VISOR to discover the location of that interaction, placed in the center of the intersection between the respective hand/glove and the interacting object. This provides an understanding about where the interaction occurs at that time step.


<p align="center" width="100%">
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000019463.jpg"> 
</p>
<p align="center" width="100%">
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000049183.jpg"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000091442.jpg"> 
</p>
<p align="center" width="100%">
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000000946.jpg"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P04_02_frame_0000005376.jpg"> 
</p>

### 2. Leverage all to the 3D

In a second stage, using Structure from Motion algorithms (COLMAP), we obtain the camera pose the global localization of the interaction in the 3D space. Running this for all the frames in the kitchen where an interaction occur, we obtain a historical distribution of all the taken actions in that kitchen. In the following image we show in blue the different camera poses, in grey the Colmap keypoints and the different locations where the interactions occur.

<p align="center" width="100%">
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/Screenshot%20from%202022-12-13%2010-31-56.png"> 
    <img width="45%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/Screenshot%20from%202022-12-13%2010-31-56.png"> 
</p>

### 3. Reproject the 3D to the 2D to obtain the affordances.

Finally, to obtain 

The dataset is !

## Baselines

We implemented different baselines, which are extensions of popular semantic segmentation datasets.
