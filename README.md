# Epic_kitchens_affordances

## Dataset creation: automatic annotations

The EPIC-Affordance dataset is a new dataset build on the Epic Kitchens 100 and Epic Kitchens VISOR. It contains **automatic annotations** generated by the intersection of both datasets. On one hand, we use the narration annotations of the Epic Kitchens 100 to obtain the semantics of the interaction (e.g "cut onion"). Then, we use the masks provided by EPIC VISOR to discover the location of that interaction, placed in the center of the intersection between the respective hand/glove and the interacting object. This provides an understanding about where the interaction occurs at that time step.

<p align="center">
<img width="800" alt="interaction_img" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg">
</p>
<p align="center" width="100%">
    <img width="30%" src="https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/P01_01_frame_0000003682.jpg"> 
    <img width="30%" src="https://i.stack.imgur.com/RJj4x.png"> 
    <img width="30%" src="https://i.stack.imgur.com/RJj4x.png"> 
</p>

In a second stage, using Structure from Motion algorithms (COLMAP), we obtain the camera pose the global localization of the interaction in the 3D space. Running this for all the frames in the kitchen where an interaction occur, we obtain a historical distribution of all the taken actions in that kitchen. In the following image we show in blue the different camera poses, in grey the Colmap keypoints and the different locations where the interactions occur.

<img width="800" alt="historical of actions" src = "https://github.com/lmur98/epic_kitchens_affordances/blob/main/imgs/Screenshot%20from%202022-12-14%2016-28-24.png">

Finally, to obtain 

The dataset is !

## Baselines

We implemented different baselines, which are extensions of popular semantic segmentation datasets.
