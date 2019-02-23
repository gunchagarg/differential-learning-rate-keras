# differential-learning-rate-keras

The phrase 'Differential Learning Rates' implies the use of different learning rates on different parts of the network.

![alt text](https://cdn-images-1.medium.com/max/1200/1*4zrt6IeIhv55mUskGhXR7Q.png)

Transfer Learning is a proven method to generate much better results in computer vision tasks. Most of the pretrained architectures (Resnet, VGG, inception, etc.) are trained on ImageNet and depending on the similarity of your data to the images on ImageNet, these weights will need to be altered more or less greatly. When it comes to modifying these weights, the last layers of the model will often need the most changing, while deeper levels that are already well trained to detecting basic features (such as edges and outlines) will need less.
