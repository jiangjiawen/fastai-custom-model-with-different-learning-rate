# fastai-custom-model-with-different-learning-rate

conv2d with padding code is from here:

https://github.com/Gasoonjia/Tensorflow-type-padding-with-pytorch-conv2d.

thanks a lot.

The core codes are:

```python
x=nn.Sequential(*list(learn.model.children()))
learn.split([x[5]])
```
They can split model into 2 layer_groups, you can check it by:

```python
print(learn.layer_groups)
```
run these scripts by

```
python3 -m torch.distributed.launch --nproc_per_node=2 testDistributeMyModel.py
```

If you have 2 gpus on your machine.
