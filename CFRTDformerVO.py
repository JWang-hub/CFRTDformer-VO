import torch
import numpy as np
import time
import torch.nn.functional as F

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from Network.VONet import VONet 


class CFRTDformerVO(object):
    def __init__(self):
        self.vonet = VONet()
        


        
        pretrained_path = 'models/ckpt_sp1.pth'
        state_dict = torch.load(pretrained_path)
        self.vonet.load_state_dict(state_dict, strict=True)



        self.vonet.cuda()

        self.test_count = 0
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 2.0 # scale factor for flow


    def test_batch(self, sample):
        self.test_count += 1
        
        
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        inputs = [img0, img1]

        self.vonet.eval()

        with torch.no_grad():
            starttime = time.time()
            pose = self.vonet(inputs)
            inferencetime = time.time()-starttime

            posenp = pose.data.cpu().numpy()
            posenp = posenp * self.pose_std # The output is normalized during training, now scale it back
            print('*******************test pose norm!!!!!!!!!!*********************')

        print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, posenp))
        return posenp

