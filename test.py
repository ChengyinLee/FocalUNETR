# read option and record the settings for exp
from click import option
from matplotlib import image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
import logging
from tqdm import tqdm
import sys
import os

from get_model import get_model
from torch.nn.modules.loss import CrossEntropyLoss
from utils.utils import DiceLoss
from utils.inference import inference_online
from monai.utils import set_determinism

# read option and record the settings for exp
import shutil
from datetime import datetime
import yaml
from utils.options import ordered_yaml

# The only changes you should make is here the path for the yml file
ppath = './options/focalunetr_private.yml'
with open(ppath, mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])

model_epoch = 13
cuda_idx = 7
test_folder = './run/exp_20220608-11:56:21' # as an example

save_test_log_path = test_folder + f"/epoch{model_epoch}_test_log_t190.txt"
assert not os.path.exists(save_test_log_path)
logging.basicConfig(filename=save_test_log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



model_dict_path = test_folder + f'/epoch_{model_epoch}.pth'
ppath = test_folder + '/train_focalnetunetr.yml'
with open(ppath, mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])


from datasets.dataset import ProstateDataset
db_test = ProstateDataset(base_dir=opt['datasets']['test']['base_dir'], 
                         list_dir=opt['datasets']['test']['list_dir'], 
                         split=opt['datasets']['test']['split'],
                         transform=None
                         )   
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
model = get_model(model_name='focalunetr')
model.load_state_dict(torch.load(model_dict_path))
model.eval()
with torch.no_grad():
    performance, mean_hd95 = inference_online(db_test, model, 
                                                num_classes=opt['network']['out_chans'], 
                                                img_size=opt['datasets']['train']['output_size'][0],  # [224, 224] -> 224
                                                logging=logging, test_save_path=None, z_spacing=1.5)
    print(f"Testing is finished!")
