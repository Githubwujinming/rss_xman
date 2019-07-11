import torch
from models.Siam_unet import SiamUNet, SiamUNetU
from models.deeplabv3 import DeepLabV3
from torch.autograd import Variable
import utils.dataset_deeplab as my_dataset
import cv2
import numpy as np
import config.rssia_config as cfg
import os
import preprocessing.transforms as trans
from torch.utils.data import DataLoader
from preprocessing.crop_img import splitimage
from PIL import Image

# def img_process(img1, img2, lbl, flag='test'):
#     img1 = img1[:, :, ::-1]  # RGB -> BGR
#     img1 = img1.astype(np.float64)
#     img1 -= cfg.T0_MEAN_VALUE
#     img1 = img1.transpose(2, 0, 1)
#     img1 = torch.from_numpy(img1).float()
#     img2 = img2[:, :, ::-1]  # RGB -> BGR
#     img2 = img2.astype(np.float64)
#     img2 -= cfg.T1_MEAN_VALUE
#     img2 = img2.transpose(2, 0, 1)
#     img2 = torch.from_numpy(img2).float()
#     if flag != 'test':
#         lbl = np.expand_dims(lbl, axis=0)
#         lbl = torch.from_numpy(np.where(lbl > 128, 1.0, 0.0)).float()
#     return img1,img2
def prediction( weight):

    best_metric = 0
    train_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    val_transform_det = trans.Compose([
        trans.Scale(cfg.TRANSFROM_SCALES),
    ])
    test_transform_det = trans.Compose([
        trans.Scale((960,960)),
    ])
    model = DeepLabV3(model_id=1,project_dir=cfg.BASE_PATH)
    # model=torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weight).items()})
    # model.load_state_dict(torch.load(weight))
    checkpoint = torch.load(weight)
    model.load_state_dict(checkpoint['state_dict'])

    # test_data = my_dataset.Dataset(cfg.TEST_DATA_PATH, '',cfg.TEST_TXT_PATH, 'test', transform=True, transform_med=test_transform_det)
    test_data = my_dataset.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH,cfg.VAL_TXT_PATH, 'val', transform=True, transform_med=test_transform_det)
    # test_data = my_dataset.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH,cfg.TRAIN_TXT_PATH, 'train', transform=True, transform_med=test_transform_det)
    test_dataloader = DataLoader(test_data, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    crop = 0

    for batch_idx, val_batch in enumerate(test_dataloader):
        model.eval()
        #
        # batch_x1, batch_x2, _, filename, h, w, green_mask1, green_mask2 = val_batch
        batch_det_img, _, filename, h, w,_,green_mask2 = val_batch
        # green_mask1 = green_mask1.view(output_w, output_h, -1).data.cpu().numpy()
        filename = filename[0].split('/')[-1].replace('image','mask_2017')
        if crop:
            pass
            # outputs = np.zeros((cfg.TEST_BATCH_SIZE,1,960, 960))
            #
            # while (i + w // rows <= w):
            #     j = 0
            #     while (j + h // cols <= h):
            #         batch_x1_ij = batch_x1[0, :, i:i + w // rows, j:j + h // cols]
            #         batch_x2_ij = batch_x2[0, :, i:i + w // rows, j:j + h // cols]
            #         # batch_y_ij = batch_y[batch_idx,: , i:i + w // rows, j:j + h // cols]
            #         batch_x1_ij = np.expand_dims(batch_x1_ij, axis=0)
            #         batch_x2_ij = np.expand_dims(batch_x2_ij, axis=0)
            #         batch_x1_ij, batch_x2_ij = Variable(torch.from_numpy(batch_x1_ij)).cuda(), Variable(
            #             torch.from_numpy(batch_x2_ij)).cuda()
            #         with torch.no_grad():
            #             output = model(batch_x1_ij, batch_x2_ij)
            #         output_w, output_h = output.shape[-2:]
            #         output = torch.sigmoid(output).view(-1, output_w, output_h)
            #
            #         output = output.data.cpu().numpy()  # .resize([80, 80, 1])
            #         output = np.where(output > cfg.THRESH, 255, 0)
            #         outputs[0, :, i:i + w // rows, j:j + h // cols] = output
            #
            #         j += h // cols
            #     i += w // rows
            #
            #
            # if not os.path.exists('./change'):
            #     os.mkdir('./change')
            # print('./change/{}'.format(filename))
            # cv2.imwrite('./change/crop_{}'.format(filename), outputs[0,0,:,:])
        else:
            batch_det_img = Variable(batch_det_img).cuda()
            with torch.no_grad():
                outputs = model(batch_det_img)

            output_w, output_h = outputs[0].shape[-2:]

            # green_mask2 = green_mask2.view(output_w, output_h, -1).data.cpu().numpy()

            output = torch.sigmoid(outputs).view(output_w, output_h, -1).data.cpu().numpy()
            # print(output.min(),output.max())
            output = np.where((output  > cfg.THRESH) , 255, 0)
            if not os.path.exists('./change'):
                os.mkdir('./change')

            print('./change/{}'.format(filename))
            cv2.imwrite('./change/{}'.format(filename), output)
if __name__ == "__main__":

    # weight="weights/CE_loss/model_tif_last.pth"
    # weight="weights/model20_1.pth"
    weight="weights/CE_loss/model_tif_deeplab18_bce_240*240_50.pth"
    prediction(weight)