import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

from utils import *

def class_saliency_map(input_filepath, model_name, class_name_index):
    if check_model_support(model_name):
        image = load_image(input_filepath)
        model = load_model(model_name)

        image = preprocess_image(image)
        image = image.unsqueeze(0)
        image.requires_grad = True

        # label tensor
        y = torch.cuda.LongTensor([label_index])

        # put model,on GPU
        model = model.cuda()

        # forward pass
        scores = model(image.cuda())

        # get non-normalized probabilities
        scores = scores.gather(1, y.view(-1, 1)).squeeze()

        # backward pass
        scores.backward()

        # saliency maps
        saliency_maps = image.grad.data
        saliency_maps = saliency_maps.abs()
        saliency_maps, _ = torch.max(saliency_maps, dim=1)

        image        = image.data.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
        image        = post_process(image)
        saliency_map = saliency_map.squeeze(0).cpu().detach().numpy()
        saliency_map = post_process(saliency_map)
        
        show_saliency_maps(image, saliency_map)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize CNN')
    
    parser.add_argument('-input_path', help='Path to input image')     
    parser.add_argument('-label', help='Target Name')
    parser.add_argument('-model', help='Model to use') 
    parser.add_argument('-o', help='Operation to be performed')

    args = parser.parse_args()

    if args.o == 'class_saliency':
        input_filepath = args.input_path
        model_name     = args.model
        class_name     = args.label

        class_name_index = class_name_to_index(class_name)

        if class_name_index == -1:
            raise ValueError('Class name does not exist.')

        class_saliency_map(input_filepath, model_name, class_name_index)    

