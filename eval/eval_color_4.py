# Code to produce colored segmentation output of panoramas in Pytorch 
# With special padding and upsampling, use 4 feature models and 1 fusion model
# April 2019
# Kailun Yang
#######################

from distutils.ccompiler import gen_preprocess_options
import numpy as np
import torch
import os
import importlib
import cv2

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage

import dataset as ds
from erfnet_pspnet_splits4 import Net
from transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
SIZE=(1024,1024)
image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((SIZE[0],SIZE[1]*4),Image.BILINEAR), #here, the resolution has been expanded to 4 times, corresponding to 4 feature models
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225])
])

def save_pred(inp, sv_path, name, image=None, alpha=0.5):
    save_img = Image.fromarray(inp)
    if image is not None:
        image = Image.fromarray(image)
        image = image.resize(save_img.size)
        save_img = Image.blend(image.convert('RGBA'), save_img.convert('RGBA'), alpha=alpha)
    save_img.save(os.path.join(sv_path, f'{name}.png'))

def main(args):
    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    dataset = getattr(ds, args.dataset)
    #gt_folder = f"/media/hieu/DATA/pano-segmentation/footpath/Mask images"
    gt_folder = None
    dataset = dataset(args.datadir, input_transform_cityscapes, subset=args.subset, gt_folder=gt_folder)
    NUM_CLASSES = dataset.NUM_CLASSES
    cmap = dataset.color_map
    loader = DataLoader(dataset,
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    global SIZE

    weightspath = args.loadDir + args.loadWeights

    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES, size=SIZE)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()


    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        
        for a in own_state.keys():
            print(a)
        for a in state_dict.keys():
            print(a)
        print('-----------')
        
        for name, param in state_dict.items():
         
            if name not in own_state:
                 continue
          
            own_state[name].copy_(param)
        
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()
    folder = f"./save_color/{args.dataset}"
    os.makedirs(folder, exist_ok=True)
    with torch.no_grad():
        for step, data in enumerate(loader):
            if gt_folder is not None:
                images, gt, filename, shape = data
            else:
                images, filename, shape = data
            shape = shape[0].item(), shape[1].item()
            images_A = images.cuda()
            outputsA = model(images_A)
            
            outputs = outputsA
            label = outputs[0].cpu().max(0)[1].data.byte()
            label_color = Colorize(dataset=args.dataset, cmap=cmap)(label.unsqueeze(0))
            label = label_color.numpy().transpose(1,2,0)
            label = cv2.resize(label, shape, interpolation = cv2.INTER_NEAREST)

            ## GT
            if gt_folder is not None:
                gt_ = gt[0][0].cpu()
                gt_color = Colorize(dataset=args.dataset, cmap=cmap)(gt_.unsqueeze(0))
                gt_ = gt_color.numpy().transpose(1,2,0)
                save_pred(gt_, folder, f"{filename[0]}_gt", image=image, alpha=0.4)

            image = images[0].numpy()
            image = (image*255).transpose(1,2,0).astype(np.uint8)
            save_pred(label, folder, filename[0], image=image, alpha=0.4)
            # label_save = ToPILImage()(label_color)
          
            # label_save.save(filenameSave) 
          
            print(step, filename[0])

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfpspnet.pth")
    parser.add_argument('--loadModel', default="erfnet_pspnet_splits4.py")
    parser.add_argument('--subset', default="pass")  #can be pass, val, test, train, demoSequence

    parser.add_argument('--datadir', default="../dataset/")
    parser.add_argument('--dataset', default="cityscapes")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
