from .utils import test_single_volume_online
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def inference_online(db_test, model, num_classes, img_size, logging, test_save_path=None, z_spacing=1.5):
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume_online(image, label, model, classes=num_classes, patch_size=[img_size, img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f assd %f' 
                     % (i_batch, case_name, 
                        np.mean(metric_i, axis=0)[0], 
                        np.mean(metric_i, axis=0)[1],
                        np.mean(metric_i, axis=0)[2]
                        ))
    metric_list = metric_list / len(db_test)
    for i in range(1, num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f assd %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    assd = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f assd : %f' % (performance, mean_hd95, assd))
    return performance, mean_hd95, assd