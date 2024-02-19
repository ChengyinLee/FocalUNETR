from click import option
from matplotlib import image
from torchvision import transforms
from datasets.dataset_with_contour import ProstateDataset, RandomGenerator
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
from torch.nn import MSELoss
from utils.inference import inference_online
from monai.utils import set_determinism

# read option and record the settings for exp
import shutil
from datetime import datetime
import yaml
from utils.options import ordered_yaml 

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# The only changes you should make is here the path for the yml file
ppath = './options/focalunetr_multi-task_with_contour.yml'
with open(ppath, mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])
        
# genarate a fodler to save running experimental setting and results
snapshot_path = './run_inhouse/focalunetr_multi-task_with_contour/' + 'exp_'+datetime.now().strftime("%Y%m%d-%H_%M_%S")
assert os.path.exists(snapshot_path) == False
os.makedirs(snapshot_path)
shutil.copy(src=ppath, dst=snapshot_path) # copy the yml file to the snapshot_path


logging.basicConfig(filename=snapshot_path + "/training.log", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# genarate the datasets for train and validaiton (or testing)
db_train = ProstateDataset(base_dir=opt['datasets']['train']['base_dir'], 
                           list_dir=opt['datasets']['train']['list_dir'], 
                           split=opt['datasets']['train']['split'],
                           transform=transforms.Compose(
                            [RandomGenerator(
                                output_size=opt['datasets']['train']['output_size'], 
                                rotate_or_flip_prob=opt['datasets']['train']['rotate_or_flip_prob'])]))
db_val = ProstateDataset(base_dir=opt['datasets']['val']['base_dir'], 
                         list_dir=opt['datasets']['val']['list_dir'], 
                         split=opt['datasets']['val']['split'], 
                         transform=None)


# create data loader
loader_train = DataLoader(db_train, batch_size=opt['train']['batch_size']['train'], shuffle=True, num_workers=8) # maybe 24 or 12


# define the model
set_determinism(seed=opt['manual_seed'])

device = torch.device(f"{opt['train']['device']}" if torch.cuda.is_available() else "cpu")

model = get_model(model_name='focalunetr').to(device)


# some train options
if opt['train']['loss_type']=="DiceCE":
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=opt['network']['out_chans'])
else:
    ce_loss = None
    dice_loss = None
    
mse_loss = MSELoss()

if opt['train']['optim']['type'] == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=opt['train']['optim']['lr'], 
                                betas=opt['train']['optim']['betas'], 
                                weight_decay=opt['train']['optim']['weight_decay'])
elif opt['train']['optim']['type'] == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=opt['train']['optim']['lr'], 
                                  weight_decay=opt['train']['optim']['weight_decay'])
elif opt['train']['optim']['type'] == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=opt['train']['optim']['lr'], 
                                momentum=opt['train']['optim']['momentum'], 
                                weight_decay=opt['train']['optim']['weight_decay']) 
else:
    optimizer = None
    assert optimizer is not None,  "No optimizer found"

assert opt['train']['scheduler'] == "e_decay"
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# writing to tensorboard
writer = SummaryWriter(snapshot_path)

max_epoch = opt['train']['max_epoch']
max_iterations = max_epoch * len(loader_train)
logging.info(f"{len(loader_train)} iterations per epoch. {max_iterations} max iterations ")

# The training 
iter_num = 0
best_performance = -1
best_hd95 = 1000
base_lr = opt['train']['optim']['lr']

for epoch in range(max_epoch):
    # train
    model.train()
    epoch_loss = 0
    for idx, batch in enumerate(loader_train):
        image_batch, label_batch, contour_batch = batch['image'].to(device), batch['label'].to(device), batch['contour'].unsqueeze(1).to(device)
        out = model(image_batch)
        outputs, contours = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss_mse = mse_loss(contours, contour_batch)
        loss = 0.5 * loss_ce + 0.5 * loss_dice + 0.5*loss_mse
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        lr_ = scheduler.get_last_lr()
        iter_num = iter_num + 1
        writer.add_scalar('info/lr', lr_, iter_num)
        writer.add_scalar('info/total_loss', loss, iter_num)
        writer.add_scalar('info/loss_ce', loss_ce, iter_num)
        writer.add_scalar('info/loss_dice', loss_dice, iter_num)
        writer.add_scalar('info/loss_mse', loss_mse, iter_num)
        logging.info('epoch %d, iteration %d, loss %f' % (epoch, iter_num, loss.item()))

        if iter_num % 5000 == 0:
            image = image_batch[1, 0:1, :, :]
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image, iter_num)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
            labs = label_batch[1, ...].unsqueeze(0) * 50
            writer.add_image('train/GroundTruth', labs, iter_num)
    epoch_loss /= len(loader_train)
    logging.info(f'epoch: {epoch}, epoch_loss:{epoch_loss}')
    writer.add_scalar('info/epoch_lr', lr_, epoch)
    writer.add_scalar('info/epoch_loss', epoch_loss, epoch)
    scheduler.step()
    
    # evaluation and saving checking point
    model.eval()
    with torch.no_grad():
        val_freq = opt['val']['val_freq']
        if epoch > 10:
            save_model_path = os.path.join(snapshot_path, 'epoch_' + str(epoch) + '.pth') # save model
            torch.save(model.state_dict(), save_model_path)
            
            if  (epoch + 1) % val_freq == 0 or (epoch >= max_epoch - 1):          
                performance, mean_hd95 = inference_online(db_val, model, 
                                                        num_classes=opt['network']['out_chans'], 
                                                        img_size=opt['datasets']['train']['output_size'][0],
                                                        logging=logging, test_save_path=None, z_spacing=1.5)
                writer.add_scalar('info/mean_dsc', performance, epoch)
                writer.add_scalar('info/mean_hd95', mean_hd95, epoch)
                
                if best_performance <= performance:
                    best_performance = performance
                    logging.info(f"We saved a better model at epoch: {epoch}, with a better DSC: {best_performance}")
                elif best_hd95 > mean_hd95:
                    best_hd95 = mean_hd95
                    logging.info(f"We saved a better model at epoch: {epoch}, with a better hd95: {best_hd95}") 
writer.close()
