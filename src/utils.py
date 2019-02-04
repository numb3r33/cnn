import requests
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models

from config import *

def class_name_to_index(class_name):
    labels = {int(key):value for (key, value) in requests.get(LABELS_URL).json().items()}
    
    for k, v in labels.items():
        if v in class_name.split(', '):
            return k
    
    return -1

def load_image(filename):
    if os.path.exists(filename):
        raise FileNotFoundError('No such file exists.')
    
    return Image.open(filename)

def preprocess_image(image):
    normalize  = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    normalize
                ])
    
    return preprocess(image)

def check_model_support(model_name):
    if model_name not in ['vgg16', 'resnet101']:
        raise ValueError('Don"t support this model type')
    
    return True

def load_model(model_name):
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    return model

def post_process(img):
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)

    return img


def show_saliency_maps(image, saliency_map):
    N = 1 # number of images
    
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency_map, cmap = plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    
    plt.show()