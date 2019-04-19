# fastai-custom-model-with-different-learning-rate

conv2d with padding code is from here:

https://github.com/Gasoonjia/Tensorflow-type-padding-with-pytorch-conv2d.

thanks.

The core codes are:

```python
x=nn.Sequential(*list(learn.model.children()))
learn.split([x[5]])
```
They can split model into 2 layer_groups, you can check it by:

```python
print(learn.layer_groups)
```
