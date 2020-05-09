from pathlib import Path
import numpy as np
from PIL import Image
import os 

def split_instance(images_dir, masks_dir, imset, dest_dir, year = '2017', type = 'train'):
    try:
        os.makedirs(os.path.join(dest_dir, 'ImageSets', year))
    except:
        pass
    print('Generating seperate mask..')
    palette = None 
    for seq_name in Path(masks_dir).iterdir():
        palette = Image.open(seq_name / '00000.png').getpalette()
        break
    with open(imset, "r") as lines:
        f = open(os.path.join(dest_dir,'ImageSets/', year,type + '.txt'), 'w')
        for line in lines:
            seq_name = line.rstrip('\n')
            print(seq_name)
            img_ids = []
            msk_ids = []
            for img in sorted((Path(images_dir) / seq_name).iterdir()):
                img_ids.append(str(img))
            for id, img in enumerate(sorted((Path(masks_dir) / seq_name).iterdir())):
                msk_ids.append(str(img))
            n = len(img_ids)
            for i in range(n):
                #print(seq_name, i)
                try:
                    mask = np.array(Image.open(msk_ids[i]))
                except:
                    mask = None
                im = Image.open(img_ids[i])
                # extract certain classes from mask (e.g. cars)
                if (i == 0):
                    class_values = np.unique(mask)
                    print(class_values)
                    for v in range(1, len(class_values)):
                        print(seq_name + '_' + str(v).zfill(2), file = f)
                    print(seq_name,i, '/', n, 'with',len(class_values) - 1,'instance(s).') 
                for v in class_values[1:]:
                    path = str(Path(dest_dir) / "Annotations" / "480p" / (seq_name + '_' + str(v).zfill(2)))
                    path_im = str(Path(dest_dir) / "JPEGImages" / "480p" / (seq_name + '_' + str(v).zfill(2)))
                    if i == 0:
                        os.system('rm -rf ' + path_im)
                        os.system('rm -rf ' + path)
                        os.makedirs(path_im)
                        os.makedirs(path)
                    file_name = str(path) + '/' + str(i).zfill(5) + '.png'
                    file_name_im = str(path_im) + '/' + str(i).zfill(5) + '.jpg'
                        
                    if mask is not None:
                        x = (mask == v)
                        x[x == True] = 1
                        x[x == False] = 0
                        img = Image.fromarray(x.astype(np.uint8))
                        img.putpalette(palette)
                        img.save(file_name)
                    #print(file_name_im)
                    im.save(file_name_im)
        f.close();
    print('\nDone...')
    #/content/DataTestChallenge/DAVIS