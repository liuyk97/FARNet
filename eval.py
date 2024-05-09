import numpy as np

from Dataloader.CD_dataset import LEVID_CDset, CDD_set
from models.CD_Net import CD_Net
from Dataloader.dataset import *
from utils.metrics import runningScore, averageMeter, RunningMetrics_CD
import tqdm
import cv2


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    params = {'alpha': 0.2, 'beta': 1.4, 'theta': 1.8, 'dataset': 'CDD', 'channel': 256, 'out_channel': 64,
              'dilation': [True, True, True], 'MFDS': True, 'CFI': True, 'CMR': True, 'adv': True, 'cos': True}
    vis_path = '/data/sdu08_lyk/data/LEVIR-CD_256x256/vis_TTT_FFFFF/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    restore_from = '/home/liuyikun/RS_segmentation_pipeline/logs/CDD/1234/best_model_fcn_1234.pkl'
    # test_set = LEVID_CDset(mode='test')
    # print("get {} images from LECID-CD test set".format(len(test_set)))
    test_set = CDD_set(mode='test')
    print("get {} images from CDD test set".format(len(test_set)))
    test_loader = DataLoader(test_set, batch_size=1, num_workers=2, shuffle=False, sampler=None)
    iters = len(test_loader) * 200
    model = CD_Net(params=params, restore_from=restore_from)
    running_metrics_val = RunningMetrics_CD(num_classes=2)
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for idx, val_data, in enumerate(tqdm.tqdm(test_loader)):
            image_A = val_data['A']
            image_B = val_data['B']
            label = val_data['L']
            image_A = image_A.cuda()
            image_B = image_B.cuda()
            label = label.cuda()
            out_A, out_B, result = model.basenet(image_A, image_B)
            out = result.max(1)[1].detach().cpu().numpy()
            # out = FillHole(out)
            lbl = label.data.detach().cpu().numpy()
            running_metrics_val.update(lbl, out)
            # cv2.imwrite(vis_path + val_data['path'][0], out.squeeze() * 255)
        score = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)
