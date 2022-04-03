# Investigating Techniques to Improve CNN Generalization on OOD Samples 

This repository is heavily inspired by the original work from the following repo:  
https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_combinations  
  
The original work is done by Madan et. al. We'll be using the original code provided by them as the baseline on which we'll make some modifications to investigate how we can improve on CNN generalization on OOD Samples.  

Initial playground code:  
This is located in demos/Deep_Learning_Project_Code_Playground.ipynb  
(This code is our initial attempt at emulating and rotating the MNIST Handwriting dataset with rotations and noise by training a simpler network than what we have in the original paper).

MNIST clothing rotation and data augmentation:
This code generates rotated MNIST clothing images from the MNIST clothing dataset, which we can easily use to experiment on since the naming format is similar to the way the original authors name their files. The MNIST clothing dataset has 10 classes (similar to MNIST handwriting). Run data/mnist_clothing/gen_mnist_clothing_discrete_rotation_data.py to auto generate the clothing dataset with rotation. The way the code work is it rotates the clothing dataset in discrete angles between 0 to 360 (in steps of 36) in order to form 10 unique rotation viewpoint classes. After running it, you can try running demos/increasing_in_distribution_combinations_modified_with_mnist_clothing.ipynb to see how the network behaves when running on the rotated clothing dataset. 
  
Contributers:  
Zixiang Loh zl3021@columbia.edu  
Alexandre Raeval atr2122@columbia.edu  
John Blackwelder jwb2168@columbia.edu  
