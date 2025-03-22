import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .model import BiSeNet

CHECKPOINT = 'face_parsing/cp/79999_iter.pth'
net = None

def vis_parsing_maps(im, parsing_anno, stride):
    part_colors = [[0, 0, 0],  # bg
                   [255, 255, 255],  # skin
                   [255, 255, 255],  # l_brow
                   [255, 255, 255],  # r_brow
                   [255, 255, 255],  # l_eye
                   [255, 255, 255],  # r_eye
                   [255, 255, 255],  # eye_g
                   [0, 0, 0],  # l_ear
                   [0, 0, 0],  # r_ear
                   [0, 0, 0],  # ear_r
                   [255, 255, 255],  # nose
                   [255, 255, 255],  # mouth
                   [255, 255, 255],  # u_lip
                   [255, 255, 255],  # l_lip
                   [0, 0, 0],  # neck
                   [0, 0, 0],  # neck_l
                   [0, 0, 0],  # cloth
                   [0, 0, 0],  # hair
                   [0, 0, 0],  # hat
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_parsing_anno_color = Image.fromarray(vis_parsing_anno_color) 
    
    return vis_parsing_anno_color

def get_face_mask(pil_img):
    global net
    
    if net is None:
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        net.load_state_dict(torch.load(CHECKPOINT))
        net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        origw, origh = pil_img.size
        
        image = pil_img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        print(np.unique(parsing))

        mask = vis_parsing_maps(image, parsing, stride=1)
        mask = mask.resize((origw, origh), Image.BILINEAR)

    return mask