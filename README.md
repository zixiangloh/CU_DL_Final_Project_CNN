# Investigating Techniques to Improve CNN Generalization on OOD Samples  

This repository is heavily inspired by the original work from the following repo:  
https://github.com/Spandan-Madan/generalization_to_OOD_category_viewpoint_combinations  
  
The original work is done by Madan et. al. We'll be using the original code provided by them as the baseline on which we'll make some modifications to investigate how we can improve on CNN generalization on OOD Samples.  
  
There are 3 datasets that we use in this code:  
1\. MNIST Clothing Rotation (To experiment on a deformable object dataset as well as discrete rotations and noisy rotations).  
<a href="docs/images/"><img src="docs/images/mnist_clothing_samples.gif" width="55"></a>  
2\. Blurred Biased Car (To increase the transformations on the cars images).  
<a href="docs/images/"><img src="docs/images/biased_cars_blurred_samples.gif" width="150"></a>  
3\. Sheared Biased Car (To increase the transformations on the cars images).  
<a href="docs/images/"><img src="docs/images/biased_cars_sheared_samples.gif" width="150"></a>  

Initial playground code:  
This is located in <a href="demos/Deep_Learning_Project_Code_Playground.ipynb">demos/Deep_Learning_Project_Code_Playground.ipynb</a>  
(This code is our initial attempt at emulating and rotating the MNIST Handwriting dataset with rotations and noise by training a simpler network than what we have in the original paper). The goal is to illustrate the difficulty when it comes to viewpoint and category learning.  

**MNIST clothing rotation and discrete data transformation**:  
This code generates rotated MNIST clothing images from the MNIST clothing dataset. The MNIST clothing dataset has 10 classes (similar to MNIST handwriting).  
1. Run <a href="utils/download_mnist_rotation.sh">utils/download_mnist_rotation.sh</a>  
2. Run <a href="data/mnist_clothing/gen_mnist_clothing_discrete_rotation_data.py">data/mnist_clothing/gen_mnist_clothing_discrete_rotation_data.py</a> to auto generate the clothing dataset with discrete rotation. The way the code work is it rotates the clothing dataset in discrete angles between 0 to 360 (in steps of 36) in order to form 10 unique rotation viewpoint classes. 
3. After running the above, you can try running <a href="demos/increasing_in_distribution_combinations_modified_with_mnist_clothing.ipynb">demos/increasing_in_distribution_combinations_modified_with_mnist_clothing.ipynb</a> to see how the network behaves when running on the rotated clothing dataset. 
4. Alternately, running the demo code directly in colab (using the colab button) automatically runs both steps 1 and 2 for you.     
  
**MNIST clothing rotation and noisy data transformation**:  
Another code generates rotated MNIST Clothing images but this time with noise. The way the noise is generated is that it adds some fraction of rotation angle on top of the discrete angles from before.  
1. Run <a href="utils/download_mnist_rotation.sh">utils/download_mnist_rotation.sh</a> if you haven't already.  
2. Run the code to generate the noisy version in <a href="/data/mnist_noisy_rotation_clothing/gen_mnist_clothing_noisy_rotation_data.py">/data/mnist_noisy_rotation_clothing/gen_mnist_clothing_noisy_rotation_data.py</a> to autogenerate the noisy rotation version.  
3. After running the above, you can try running <a href="demos/increasing_in_distribution_combinations_modified_with_mnist_noisyrotclothing.ipynb">demos/increasing_in_distribution_combinations_modified_with_mnist_noisyrotclothing.ipynb</a> to see the results of running on a noisy rotation dataset.  
4. Alternately, running the demo code directly in colab (using the colab button) automatically runs both steps 1 and 2 for you.    
  
Diagramatic explanation for the MNIST Clothing data architecture:  
<a href="docs/images/"><img src="docs/images/MNIST_Clothing_Data_Architecture.png" width="900"></a>  

Biased car with blurring transformation:
This code transforms the original biased car dataset with additional blurring effect. Please run the bash script in utils/download_biased_cars.sh first to download the original biased cars dataset and have it decompressed and setup. And then run data/biased_cars_blurred/generate_biased_cars_blurred.py which transforms the original data and generates the blurred bias cars dataset. After that, you can try running demos/using_biased_cars_modified_with_blur.ipynb to train and evaluate the biased cars results. There's no colab button for this code unlike the MNIST clothing version so everything has to be run separately.  
  
Biased car with shearing transformation:
This code transforms the original biased car dataset with additional shearing effect. Please run the bash script in <a href="utils/download_biased_cars.sh">utils/download_biased_cars.sh</a> first to download the original biased cars dataset and have it decompressed and setup. And then run <a href="data/biased_cars_sheared/generate_biased_cars_sheared.py">data/biased_cars_sheared/generate_biased_cars_sheared.py</a> which transforms the original data and generates the sheared bias cars dataset. After that, you can try running <a href="demos/using_biased_cars_modified_with_shear.ipynb">demos/using_biased_cars_modified_with_shear.ipynb</a> to train and evaluate the biased cars results. There's no colab button for this code unlike the MNIST clothing version so everything has to be run separately. 
  
Diagramatic explanation for the Blurred/Sheared biased cars data architecture:  
<a href="docs/images/"><img src="docs/images/Biased_Cars_Blurred_Sheared_Data_Architecture.png" width="900"></a>  
  
We generated 4 models to evaluate different models:  
1. <a href="res/models/LATE_BRANCHING_COMBINED_no_batch_norm.py">LATE_BRANCHING_COMBINED_no_batch_norm</a>  
2. <a href="res/models/LATE_BRANCHING_COMBINED_low_momentum.py">LATE_BRANCHING_COMBINED_low_momentum</a>  
3. <a href="res/models/LATE_BRANCHING_COMBINED_avg_momentum.py">LATE_BRANCHING_COMBINED_avg_momentum</a>  
4. <a href="res/models/LATE_BRANCHING_COMBINED_high_momentum.py">LATE_BRANCHING_COMBINED_high_momentum</a>  
  
The way to run these models is to change the ARCH parameter in the demo code to the names above.  
  
To run with late-stopping, we can just change the epoch to a longer value than the default of 5.  
  
Contributers:  
Zixiang Loh zl3021@columbia.edu  
Alexandre Raeval atr2122@columbia.edu  
John Blackwelder jwb2168@columbia.edu  
