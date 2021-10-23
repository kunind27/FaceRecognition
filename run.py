# -*- coding: utf-8 -*-
"""run.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mPpmUcmsYg4M5D4SoRpf69MWZvL1YFjZ
"""

from google.colab import drive
drive.mount("/content/gdrive")

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

PATH_OF_DATA= '/content/gdrive/"My Drive"/ES_FaceMatch_Dataset'
!ls {PATH_OF_DATA}

ROOT_DIR = 'gdrive/My Drive/ES_FaceMatch_Dataset'
IMG_DIR = 'dataset_images'
TRAIN_DIR = 'train.csv'
TEST_DIR = 'test.csv'
ZIP_FILE_DIR = '/content/gdrive/My Drive/images.zip'

import os 
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_test = pd.read_csv(os.path.join(ROOT_DIR, TEST_DIR))
df_test

!pip install facenet-pytorch

"""# Cropping the Images using MTCNN"""

image_names_list = pd.unique(df_test[['image1', 'image2']].values.ravel())
print(f"No. of unique images = {len(image_names_list)}")

from facenet_pytorch import MTCNN
mtcnn = MTCNN()

from PIL import Image
from skimage.transform import resize
from tqdm.notebook import tqdm

IMG_ROOT_DIR = os.path.join(ROOT_DIR, IMG_DIR)
CROPPED_IMG_DIR = '/content/cropped'

for image_name in tqdm(image_names_list, desc = "No. of Images"):
    # Crop all the images present in the test set
    image = Image.open(os.path.join(IMG_ROOT_DIR,image_name)).convert("RGB")
    save_path = os.path.join(CROPPED_IMG_DIR, image_name)
    cropped_image = mtcnn(image, save_path = save_path)

    # Checks if MTCNN could find the face in the image or not
    if cropped_image is None:
        cropped_image = (resize(np.array(image), (160,160), anti_aliasing = True)*255).astype('uint8')
        cropped_image = Image.fromarray(cropped_image)
        cropped_image.save(os.path.join(CROPPED_IMG_DIR, image_name))

# Play an audio beep. Any audio URL will do.
from google.colab import output
output.eval_js('new Audio("https://upload.wikimedia.org/wikipedia/commons/0/05/Beep-09.ogg").play()')

"""# Using DataLoader to Load the Image"""

del mtcnn

from skimage import io
import skimage.transform
from torch.utils.data import DataLoader
from torchvision import transforms, utils

class imagePairsTest(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file_dir, img_dir, transform = None):
        super(imagePairsTest, self).__init__()
        
        self.csv_file = pd.read_csv(os.path.join(root_dir, csv_file_dir))
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img1_names = os.path.join(self.img_dir,self.csv_file.iloc[idx, 0])
        img2_names = os.path.join(self.img_dir, self.csv_file.iloc[idx, 1])
        
        # imread returns the numpy array of RGB values for the image. Shape = (H,W,3)
        image1 = io.imread(img1_names)
        image2 = io.imread(img2_names)
        
        sample = (image1, image2)

        if self.transform:
            sample = self.transform(sample)

        return sample

# Defining Image transforms - Image Preprocessing

# Rescaling Pair of Images together
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2 = sample[0], sample[1]

        h, w = image1.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image1 = skimage.transform.resize(image1, (new_h, new_w))
        
        
        h, w = image2.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image2 = skimage.transform.resize(image2, (new_h, new_w))

        return (image1, image2)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image1, image2 = sample[0], sample[1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h)
        left = torch.randint(0, w - new_w)

        image1 = image1[top: top + new_h, left: left + new_w]
        image2 = image2[top: top + new_h, left: left + new_w]

        return (image1, image2)

class Normalize(object):
    def __init__(self, mean = 127.5, std = 128.0):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        image1, image2 = sample[0], sample[1]
        image1 = (image1-self.mean)/self.std
        image2 = (image2-self.mean)/self.std
        return (image1, image2)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2 = sample[0], sample[1]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        return (torch.tensor(image1.copy(), dtype = torch.float32).contiguous(),
                torch.tensor(image2.copy(), dtype = torch.float32).contiguous())

testset = imagePairsTest(root_dir = ROOT_DIR, csv_file_dir= TEST_DIR,
                     img_dir = CROPPED_IMG_DIR, 
                     transform = transforms.Compose([Normalize(),ToTensor()]))

"""# Defining the Model
0. **MTCNN** - Used for Data Pre-processing, our data is "noisy", for facial recognition, we only need faces of people, but our images contain lot of background as well. This is a pre-trained intelligent algorithm, which recognizes faces in an image and intelligently crops the image to only include the facial region of a person to help us in facial recognition task and reduces the unnnecessary information in the images
1. **Pre-trained ResNet** - Our CNN Layer, takes in images as input, outputs vector embeddings of that image
2. **Neural Tensor Network (NTN)** - Takes in as input the vector embeddings of the two image pairs being compared and outputs a $K$ dimensional **similarity score vector** ($K$ is a hyperparameter) which stores raw similarity scores between the image pair. <a href = "https://proceedings.neurips.cc/paper/2013/file/b337e84de8752b27eda3a12363109e80-Paper.pdf">Neural Tensor Network paper</a>
3. **Feedforward Neural Network (FFNN)** - Vanilla Neural Networks which take in as input, the output of the NTN Layer and produces a 2 dimensional output of same person/not-same-person
"""

# Defining the Neural Tensor Network Layer as a Separate Class of Itself
class NTNLayer(torch.nn.Module):
    def __init__(self, output_layer_dim):
        """
        :param: d: Input Dimension of the NTN - i.e Dimension of the Graph/ Node Embeddings
        :param: k: Output Dimension of the NTN - No. of Similarity Scores to output
        """
        super(NTNLayer, self).__init__()
        self.d = 512 # Input Dimension of the NTN
        self.k = output_layer_dim # Output dimension of the NTN 
        self.params()
        self.initializeParams()
    
    def params(self):
        self.W = torch.nn.Parameter(torch.Tensor(self.d,self.d,self.k))
        self.V = torch.nn.Parameter(torch.Tensor(self.k, 2*self.d))
        self.b = torch.nn.Parameter(torch.Tensor(self.k,1))

    def initializeParams(self): 
        torch.nn.init.kaiming_normal_(self.W, a=0.1, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.V, a=0.1, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.b, a=0.1, nonlinearity='leaky_relu')
        
    def forward(self, h1, h2):
        """Returns 'K' Rough Similarity Scores between the Pair of Images
        The Neural Tensor Network (NTN) outputs 'K' similarity scores where 'K' is a hyperparameter
        :param: h1 : Embedding of Image 1 - (B,D)
        :param: h2 : Embedding of Image 2 - (B,D)
        """
        B = h1.shape[0]
        scores = torch.mm(h1, self.W.view(self.d, -1)) # (B,D) x (D, K*D) -> (B, K*D)
        scores = scores.view(B,self.d,self.k) # (B,K*D) -> (B,D,K)
        scores = (scores*h2.unsqueeze(-1)).sum(dim=1) # (B,D,K) * (B,D,1) -> (B,K)
        
        concatenated_rep = torch.cat((h1, h2), dim=1) # (B,2D)
        scores = scores + torch.mm(concatenated_rep, self.V.t()) # (B,2D) x (2D,K) -> (B,K)
        scores = scores + self.b.t() # (B,K) + (1,K) = (B,K)
        
        leaky_relu = torch.nn.LeakyReLU(negative_slope = 0.1)
        scores = leaky_relu(scores)
        return scores

from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F

class Flatten(torch.nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return x
class normalize(torch.nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

# Decoder Part of the Model, including the ResNet CNN Layer begins
class Decoder(torch.nn.Module):
    def __init__(self, ntn_output_dim = 128):
        super(Decoder, self).__init__()
        self.ntn_output_dim = ntn_output_dim
        self.setupLayers()

    def setupResnet(self):
        # Downloading the Pre-trained ResNet CNN Layer
        model_ft = InceptionResnetV1(pretrained='vggface2', classify=False)

        # Listing all the final layers, which we are going to train
        layer_list = list(model_ft.children())[-5:]
        model_ft = torch.nn.Sequential(*list(model_ft.children())[:-5])

        # Since we just want to train final layers
        for param in model_ft.parameters():
            param.requires_grad = False
        
        # Re-attaching the final layers back to the model - automatically sets requires_grad = True
        model_ft.avgpool_1a = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model_ft.last_linear = torch.nn.Sequential(Flatten(),
            torch.nn.Linear(in_features=1792, out_features=512, bias=False),
            normalize())
        
        return model_ft
    
    def setupLayers(self):
        # ResNet and Neural Tensor Network Layer
        self.resnet = self.setupResnet()
        self.NTN = NTNLayer(self.ntn_output_dim)
        
        # Linear Layers for the Final Output
        self.lin1 = torch.nn.Linear(self.ntn_output_dim,64)
        self.lin2 = torch.nn.Linear(64,32)
        self.lin3 = torch.nn.Linear(32,16)
        self.lin4 = torch.nn.Linear(16,8)
        self.lin5 = torch.nn.Linear(8,2)

    def FCNN(self, x):
        X = self.lin1(x)
        X = X.relu()
        X = self.lin2(X)
        X = X.relu() 
        X = self.lin3(X)
        X = X.relu()
        X = self.lin4(X)
        X = X.relu() 
        X = self.lin5(X)
        return X

    def rbfKernel(self, h1,h2):
        distance = h1-h2
        distance = torch.sum(distance*distance, dim = 1)
        return torch.exp(-distance).view(-1,1)

    def forward(self, X1, X2):
        # Passing input images through the ResNet to generate Image Embeddings
        h1 = self.resnet(X1)
        h2 = self.resnet(X2)

        # Passing the image embeddings via the NTN and the FCNN layer for predictions
        y_pred = self.NTN(h1, h2)
        #y_pred = torch.cat((y_pred, self.rbfKernel(h1,h2)), dim=1)
        y_pred = self.FCNN(y_pred)

        return y_pred

"""# Evaluating Model"""

def test_predict(loader):
    decoder.eval()
    predictions_list = []

    with torch.no_grad():
        for batch in loader:
            # Passing it via final encoder to get predictions
            y_pred = decoder(batch[0].to(device), batch[1].to(device))
            y_pred = y_pred.argmax(dim=1).cpu().detach().numpy().ravel()
            predictions_list.extend(list(y_pred))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return predictions_list

PATH_TO_MODEL = 'gdrive/My Drive/best_model.pth'
decoder = Decoder()
# Choose whatever GPU device number you want
decoder.load_state_dict(torch.load(PATH_TO_MODEL))
decoder.to(device)

# Evaluating the test results
# No shuffling since we want to append test results back to df_test
batch_size = 64
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False)
predictions_list = test_predict(testloader)
df_test['label_pred'] = pd.Series(predictions_list)

print(df_test)

df_test.to_csv('predictions.csv')