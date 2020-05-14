from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import matplotlib 

### My libs
from dataset import DAVIS_MO_Test
from model import STM

import dilate_and_gaussian_mask

torch.set_grad_enabled(False) # Volatile

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    return parser.parse_args()

args = get_arguments()

GPU = args.g
YEAR = args.y
SET = args.s
VIZ = args.viz
DATA_ROOT = args.D

# Model and version
MODEL = 'STM'
print(MODEL, ': Testing on DAVIS')

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
if torch.cuda.is_available():
    print('using Cuda devices, num:', torch.cuda.device_count())

if VIZ:
    print('--- Produce mask overaid video outputs. Evaluation will run slow.')
    print('--- Require FFMPEG for encoding, Check folder ./viz')

def Run_video(Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]
    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        
        # segment
        with torch.no_grad():
            ####### Dilate image t by mask (t - 1)
            img = Fs[0,:,t].cpu().numpy() * 255
            img = np.transpose(img, (1, 2, 0))
            new_img = dilate_and_gaussian_mask.process_input_image(mask_t_1 = np.argmax(Es[0,:,t-1].cpu().numpy(), axis=0).astype(np.uint8), jpeg_t = img)
            try:
                os.makedirs('image')
            except:
                pass
            new_img.save('image/{:05d}.jpg'.format(t))
            new_img = np.array(new_img) / 255.
            total_on = np.sum(new_img > 0)
            #print(t, total_on, new_img.shape[0] * new_img.shape[1] * 3)
            #if total_on:
            Fs[0,:,t] = torch.from_numpy(np.transpose(new_img, (2, 0, 1))).float()
            
            #######
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es[0].cpu().numpy(), Fs



Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = nn.DataParallel(STM())
if torch.cuda.is_available():
    model.cuda()
model.eval() # turn-off BN

pth_path = 'STM_weights.pth'
print('Loading weights:', pth_path)
model.load_state_dict(torch.load(pth_path))

code_name = '{}_DAVIS_{}{}'.format(MODEL,YEAR,SET)
print('Start Testing:', code_name)



from pathlib import Path
palette = None 
for seq_name in Path(DATA_ROOT + '/Annotations/480p/').iterdir():
    palette = Image.open(str(seq_name) + '/00000.png').getpalette()
    break

for seq, V in enumerate(Testloader):
    Fs, Ms, num_objects, info = V
    seq_name = info['name'][0]
    num_frames = info['num_frames'][0].item()
    print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    
    pred, Es, Fs = Run_video(Fs, Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        
    # Save results for quantitative eval ######################
    test_path = os.path.join('./test_heatmap', code_name, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
        for o in range(num_objects):
            os.makedirs(test_path + '/' +str(o + 1))
        
    print('Saving mask...')
    
    print(Es.shape)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
    
        #for o in range(num_objects):
        #    heatmap =  Es[o + 1, f]
        #    #print(f, o, np.amin(heatmap), np.amax(heatmap))
        #    matplotlib.image.imsave(os.path.join(test_path, str(o + 1), '{:05d}.png'.format(f)), heatmap)
        #    #matplotlib.image.imsave('Image.jpg', denormalize(image[0]))  
        #    #matplotlib.image.imsave('Mask.png', pr_mask[0,:,:,0], cmap = 'gray')  
    
    #for o in range(num_objects):
    #    print('Saving heatmap {}_{}'.format(seq_name, o + 1))
    ##    vid_path = os.path.join('./test_heatmap/', code_name, '{}.mp4'.format(seq_name + '_' + str(o + 1)))
    #    frame_path = os.path.join('./test_heatmap/', code_name, seq_name, str(o + 1),'%05d.png')
    #    #os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
    #    os.system('ffmpeg -i {} -c:v libx264 -nostats -vf "fps=10,format=yuv420p" {}.mp4'.format(frame_path, vid_path))
        #ffmpeg -r 1/5 -i img%03d.png -c:v libx264 -vf "fps=25,format=yuv420p" out.mp4
    # If is instance object -> change color from id 1 -> exactly id.
    
    if VIZ:
        print('Saving video...')
        from helpers import overlay_davis
        # visualize results #######################
        viz_path = os.path.join('./viz/', code_name, seq_name)
        if not os.path.exists(viz_path):
            os.makedirs(viz_path)
        
        for f in tqdm.tqdm(range(num_frames)):
            pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
            pE = pred[f]
            #x = np.amax(pE)
            #pE[pE == x] = 0
            #print(seq_name, f, [np.sum(pE == x) for x in range(num_objects + 1)])
            
            canvas = overlay_davis(pF, pE, palette)
            canvas = Image.fromarray(canvas)
            canvas.save(os.path.join(viz_path, 'f{:05d}.jpg'.format(f)))

        vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
        frame_path = os.path.join('./viz/', code_name, seq_name, 'f%05d.jpg')
        os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))



