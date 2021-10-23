# FaceRecognition
This Github repository contains code submitted as an entry for the AI Hackathon hosted by Analytics Club, IIT Bombay in collaboration with EarlySalary.

We used an end-to-end **Deep Siamese Model** to recognize faces in the dataset. The model created was based on an **encoder-decoder paradigm**, where a **Deep-CNN** layer was used to generate vector embeddings for the input image pair, and a novel decoder layer has been used to predict the similarity between the input images.

## Layer Information
#### 0. Multi-task Cascaded Convolutional Neural Networks (MTCNN)<br>
#### 1. Inception ResNet V1<br>
#### 2. Neural Tensor Network<br>
#### 3. RBF Kernel Similarity<br>
#### 4. Feedforward Neural Network

## Multi-task Cascaded Convolutional Neural Networks (MTCNN)
A pre-trained MTCNN layer has been used - implemented in `facenet_pytorch` to pre-process images by intelligently cropping only the facial region of the imaages. All those images, in which the MTCNN model could not find any faces, have simply been resized to the size `160px * 160px`.

## Inception ResNet V1
A pre-trained Inception ResNet V1 layer has been used - implemented in `facenet_pytorch` to generate vector embeddings of the input cropped images. The pre-trained model has been trained on the <a href = "https://www.robots.ox.ac.uk/~vgg/data/vgg_face/">**VGGFace2 Dataset** </a> which contains 3.31 million images of 9131 subjects. We implemented the concept of **Transfer Learning** for this layer. We freezed all but the last layer of this model, to allow finetuning of the ResNet model to better fit our Dataset.

## Neural Tensor Network
The <a href="https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf">Neural Tensor Network</a> along with the RBF Kernel similarity begin the decoder part of our model. The Neural Tensor Network has been used successfully by Bai et.al in <a href = "https://arxiv.org/pdf/1808.05689v4.pdf">**SimGNN**</a> for the problem of **Graph Similarity Computation** for calculating **Graph Edit Distances**. Additionally Neural Tensor Networks have proven their mettle in computing semantic similarity between a pair of word embeddins. This layer takes two image embeddings as an input and calculates 'K' similarity scores between the two embeddings (where K is a hyperparameter).

## RBF Kernel
The RBF Kernel similarity, is just a sanity check to supplement the similarity scores of the Neural Tensor Network. Its output is just a single value. It takes in two image embeddings and calculates the value of exponential raised to the power of the square of the L2-Norm of the differnce between the two vector embeddings.

## Feedforward Neural Network
The output from the Neural Tensor Network and the RBF Kernel are concatenated and jointly fed into the feedforward neural network layer which predicts if the two images are of the same person or not. Assuming a binary classification problem of same or not-same.
