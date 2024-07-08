import torch
from torch.autograd import Variable
import torch.nn as nn
import skimage
import skimage.io as io
from skimage.transform import rescale, resize
from torchvision import transforms
import numpy as np
import scipy.io as scio

from resample.resampling import rectification
from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    model_class = nn.DataParallel(model_class)

if torch.cuda.is_available():
    model_en = model_en.cuda()
    model_de = model_de.cuda()
    model_class = model_class.cuda()


model_en.load_state_dict(torch.load('./geoProjModels/model_en.pkl', map_location=torch.device('cpu')), strict=False)
model_de.load_state_dict(torch.load('./geoProjModels/model_de.pkl', map_location=torch.device('cpu')), strict=False)
model_class.load_state_dict(torch.load('./geoProjModels/model_class.pkl', map_location=torch.device('cpu')), strict=False)

model_en.eval()
model_de.eval()
model_class.eval()  

testImgPath = './imgs'
saveFlowPath = '.'

def scale_resize_image(image, resolution=(256, 256)):
    image = resize(image, resolution, mode='constant')
    # as toarch float32
    image = image.astype(np.float32)
    return image

correct = 0
for index, types in enumerate(['house1','house2','living_room']):
        imgPath = '%s%s%s%s' % (testImgPath, '/', types, '.jpg')
        disimgs = io.imread(imgPath)
        # save image scale
        resol = disimgs.shape
        # log resolution
        print('Resolution:')
        print(resol)
        disimgs = scale_resize_image(disimgs)
        disimgs = transform(disimgs)
        
        use_GPU = torch.cuda.is_available()
        if use_GPU:
            disimgs = disimgs.cuda()
        
        disimgs = disimgs.view(1, 3, 256, 256)
        disimgs = Variable(disimgs)
        
        middle = model_en(disimgs)
        flow_output = model_de(middle)
        clas = model_class(middle)
        
        _, predicted = torch.max(clas.data, 1)
        if predicted.cpu().numpy()[0] == index:
            correct += 1

        u = flow_output.data.cpu().numpy()[0][0]
        v = flow_output.data.cpu().numpy()[0][1]

        # transpose 3 * H * W to H * W * 3
        disimgs = disimgs.data.cpu().numpy()
        disimgs = disimgs[0].transpose(1, 2, 0)
        resImg, resMsk = rectification(disimgs, [u, v])
        # resize back
        resImg = rescale(resImg, (resol[0]/256, resol[1]/256), mode='constant')
        io.imsave('%s%s%s%s' % (saveFlowPath, '/', types, '_res.jpg'), resImg)
        #saveMatPath =  '%s%s%s' % (saveFlowPath, '/', '.mat')
        #scio.savemat(saveMatPath, {'u': u,'v': v}) 

