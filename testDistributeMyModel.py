from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.distributed import *
import argparse
from unet import Unet

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

# DATA_PATH = '/share/share/data/LTIS/INTE_DATA/TH+LTIS_datagen/data_tumor_fastai'
DATA_PATH = '../data'
def get_y_fn(x):
    return Path(str(x.parent)+'mask')/x.name

codes = array(['tumor'])
custom_stats = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

src_size=np.array([512,512])
bs,size = 9,src_size//2

src = (SegmentationItemList.from_folder(DATA_PATH).split_by_folder(valid='val').label_from_func(get_y_fn,classes=codes))
data = (src.transform(get_transforms(),size=size,tfm_y=True).databunch(bs=bs).normalize(custom_stats))

def dice(input, target):
    input = torch.sigmoid(input)
    inputs = (input > 0.5).float()
    targets = target.float()
    return 2. * (inputs * targets).sum() / (inputs.sum() + targets.sum())

metrics=dice
wd=1e-2

def criterion(input,target):
    return F.binary_cross_entropy_with_logits(input, target.float())

learn = Learner(data, Unet(3,1), loss_func=criterion, metrics=metrics, wd=wd, model_dir="mymodels").to_distributed(args.local_rank)

x=nn.Sequential(*list(learn.model.children()))
learn.split([x[5]])
learn.init(nn.init.kaiming_normal_)

lr=1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-1')
learn.load('stage-1')

learn.unfreeze()
lrs = slice(lr/100,lr)
learn.fit_one_cycle(12, lrs, pct_start=0.8)
learn.save('stage-2')

# learn=None
# gc.collect()

# size = src_size
# bs=9

# data = (src.transform(get_transforms(),size=size,tfm_y=True).databunch(bs=bs).normalize(custom_stats))
# learn = unet_learner(data, models.resnet34, loss_func=criterion, metrics=metrics, wd=wd).to_distributed(args.local_rank).load('stage-2')

# lr=7e-5
# learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

# learn.save('stage-1-big')
# learn.load('stage-1-big')

# learn.unfreeze()
# lrs = slice(lr/1000,lr/10)
# learn.fit_one_cycle(10, lrs)
# learn.save('stage-2-big')
