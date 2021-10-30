# FaceRecognition

**Link to Model** - https://drive.google.com/file/d/1RBWUOFFjPXy1w-eMFN4mtQuZlm05QaaY/view?usp=sharing.<br>Couldn't upload files more than 25 mb on Github.<br> 
This Github repository contains code submitted as an entry for the **AI Hackathon** hosted by Analytics Club, IIT Bombay in collaboration with EarlySalary.

We used an end-to-end **Deep Siamese Model** to recognize faces in the dataset. The model created was based on an **encoder-decoder paradigm**, where a **Deep-CNN** layer was used as an **encoder** to generate vector embeddings for the input image pair, and a **novel decoder layer** has been used to predict the similarity between the input images.

## File Structure
1. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/hackathon.ipynb">hackathon.ipynb</a> - A cleaned Jupyter Notebook file, containing my implementation of the algorithm, i.e training and testing.
2. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/run.ipynb">run.ipynb</a> - Jupyter Notebook consisting of code to load the model and the data and create a `csv` file of the predictions.
3. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/run.py">run.py</a> - Python version of the executable run file. Just need to change the file directory variable values to your local system.
4. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/predictions.csv">predictions.csv</a> - `csv` file containing predictions on the test set.
5. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/requirements.txt">requirements.txt</a> - Text file containing system requirements to run the code.
6. <a href = "https://github.com/kunind27/FaceRecognition/blob/main/work_summary.pdf">work_summary.pdf</a> - PDF File containing summary of the work done and detailed information about the problem modelling, the layers used and the hyperparameters.

## Problem Modelling
We modelled the problem using a Siamese Network paradigm. We took a pair of images, and passed them through the same network (that is two different networks with the same architecture with weight sharing), embedded both of them separately and tried to get a **similarity probability** as an output using the Softmax function. This modelling seemed very natural, given the format of the training data.

Additionally, there existed a **class imbalance** in the training data. The training data only had **36% positive examples** - examples of similar faces/people. Thus we used a **weighted** version of the **Cross Entropy Loss** as our objective function, to deal with this and improve the model’s performance.

The optimizer used is **AdamW** - which was proposed in the paper <a href="https://arxiv.org/abs/1711.05101">“Decoupled Weight Decay Regularization”</a>. AdamW is a modification of Adam in PyTorch, where AdamW **corrects the weight decay** issue prevalent in many momentum based optimizers (Adam is a combination of RMSProp and Momentum based Gradient Descent). Hyperparameter Settings are given in the last section.


## Layer Information
#### 0. Multi-task Cascaded Convolutional Neural Networks (MTCNN)<br>
#### 1. Inception ResNet V1<br>
#### 2. Neural Tensor Network<br>
#### 3. RBF Kernel Similarity<br>
#### 4. Feedforward Neural Network

## Multi-task Cascaded Convolutional Neural Networks (MTCNN)
A pre-trained MTCNN layer has been used - implemented in <a href="https://github.com/timesler/facenet-pytorch">`facenet_pytorch`</a> to pre-process images by intelligently cropping only the facial region of the images. All those images, in which the MTCNN model could not find any faces, have simply been resized to the size `160px * 160px`.

## Inception ResNet V1
A pre-trained Inception ResNet V1 layer has been used - implemented in <a href="https://github.com/timesler/facenet-pytorch">`facenet_pytorch`</a> to generate vector embeddings of the input cropped images. The pre-trained model has been trained on the <a href = "https://www.robots.ox.ac.uk/~vgg/data/vgg_face/">VGGFace2 Dataset</a> which contains **3.31 million** images of **9131 subjects**. We implemented the concept of **Transfer Learning** for this layer. We froze all but the last layer of this model, to allow finetuning of the ResNet model to better fit our Dataset.

## Neural Tensor Network
The <a href="https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf">Neural Tensor Network</a> along with the RBF Kernel similarity begin the decoder part of our model. The Neural Tensor Network has been used successfully by Bai et al in <a href = "https://arxiv.org/pdf/1808.05689v4.pdf">SimGNN</a> for the problem of **Graph Similarity Computation** for calculating **Graph Edit Distances**. Additionally Neural Tensor Networks have proven their mettle in computing semantic similarity between a pair of word embeddings. This layer takes two image embeddings as an input and calculates 'K' similarity scores between the two embeddings (where K is a hyperparameter).

## RBF Kernel
The RBF Kernel similarity, is just a sanity check to supplement the similarity scores of the Neural Tensor Network. Its output is just a single value. It takes in two image embeddings and calculates the value of exponential raised to the power of the square of the L2-Norm of the differnce between the two vector embeddings. Although, this layer is optional and the network gives the exact same performance even without it. Again this, is just a kind of a helper function to the NTN Layer and not the main layer.

## Feedforward Neural Network
The output from the Neural Tensor Network and the RBF Kernel are concatenated and jointly fed into the feedforward neural network layer which predicts if the two images are of the same person or not. Assuming a **binary classification** problem of same or not-same.

## Results
The model was trained on **Google Colab's Free GPU** Runtime. The model achieved a Training Acurracy of **93.55%**
