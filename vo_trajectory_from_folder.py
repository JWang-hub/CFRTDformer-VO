from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow,load_kiiti_intrinsics
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat, tartan2kitti
from evaluator.tartanair_evaluator import TartanAirEvaluator
from CFRTDformerVO import CFRTDformerVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=1216,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=320,
                        help='image height (default: 448)')
    parser.add_argument('--test-dir', default='data/kitti/10/image_2',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='data/kitti/poses/10.txt',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    testvo = CFRTDformerVO()

    # load trajectory data from a folder
    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    motionlist = []
    testname = "10"
    while True:
        try:
            sample = next(testDataiter)
        except StopIteration:
            break

        motions= testvo.test_batch(sample)
        motionlist.extend(motions)

    poselist = ses2poses_quat(np.array(motionlist))

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True)       
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
    np.savetxt('results/'+testname+'.txt',tartan2kitti(poselist))