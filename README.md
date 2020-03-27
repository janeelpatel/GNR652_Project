# GNR652 Project

## Papers Implemented

[Vanilla GAN](https://arxiv.org/abs/1406.2661) by Ian Goodfellow et al. (GAN implemented using linear fully-connected layers and non-linear activation functions, trained on MNIST to generate realistic looking images of hand-written digits)

[DCGAN](https://arxiv.org/abs/1511.06434) by Radford et al. (GAN implemented using CNNs, trained on CelebA to generate images which look similar to the training data) 

[Globally and Locally Consistent Image Completion](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwj8k9a35bXoAhUHX30KHW45Cq8QFjABegQIAxAB&url=http%3A%2F%2Fiizuka.cs.tsukuba.ac.jp%2Fprojects%2Fcompletion%2Fdata%2Fcompletion_sig2017.pdf&usg=AOvVaw21w-Qaj87fQjmeZUCke83X) Networks for image impainting tasks by Iizuka et al. for image impainting tasks.

## Globally and Locally Consistent Image Completion

The code for training and testing the network's predictions is attached [above](). For this particular implementation, the network is trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset in batches of 16 using the AdaDelta optimizer. 

### Architecture

* Completion Network : The completion network is based on a fully convolutional network whose input is an RGB image with a binary channel that indicates the image completion mask (1 for a pixel to be completed), and the output is an RGB image. The output pixels outside of the completion regions are restored to the input RGB values. The network model decreases the resolution twice, using strided convolutions to 1/4 of the original size, which is important to generate non-blurred texture in the missing regions. Dilated convolutional layers are also used in the mid-layers, allowing to compute each output pixel with a much larger input area, while still using the same amount of parameters and computational power.

* Context Discriminators (Local and Global) : The global context discriminator takes as an input the entire image rescaled to 256×256 pixels. It consists of six convolutional layers and a single fully-connected layer that outputs a single 1024-dimensional vector. The local context discriminator follows the same pattern, except that the input is a 128×128-pixel image patch centered around the completed region. In case the image is not a completed image, a random patch of the image is selected, as there is no completed region to center it on.

![Overview of the architecture for image completion learning](https://github.com/janeelpatel/GNR652_Project/blob/master/glcic_arch.png)

### Results

![](https://github.com/janeelpatel/GNR652_Project/blob/master/test1.png) ![](https://github.com/janeelpatel/GNR652_Project/blob/master/test2.png) ![](https://github.com/janeelpatel/GNR652_Project/blob/master/test3.png)

Note that for both cases, the top image is fed into the completion network, while the bottom image is the reconstructed (/inpainted) image for the given input.

Training for a very few number of epochs explains the blurry texture obtained in the impainted portion (/mask). This could be because the model has still not quite converged to its global optimum but oscillates around a sub-optimal point. The model still does a decent job when it comes to reconstructing eyes in the masked region.
