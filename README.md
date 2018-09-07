# Image inpainting model implementation in Tensorflow
'Image Inpainting for Irregular Holes Using Partial Convolutions' first tensorflow primary instance, fully implemented using tensorflow, without modifying the source code.<br>

Original paper: [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf)

Demo on YouTube: [link](https://www.youtube.com/watch?v=gg0F5JjKmhA)
## Partial Convolution
Use curr_bin_mask to represent the mask of the current binary; conved_mask represents the result of convolution of the binary mask, corresponding to sum(M) in the text; new_bin_mask represents the new binary mask after convolution, and the update rule is:
```
((conved_mask==0)==0) 
```
therefore, the local convolution is calculated as follows:
```
Pconv(x) = (Conv(x*curr_bin_mask)*conved_mask+b)*new_bin_mask
```

The operation with the new mask is to ensure that the invalid input is zero, as described in the text.

## Network structure
[U_net structure diagram:](https://arxiv.org/abs/1411.4038)<br>&#8195;![image](https://github.com/Rongpeng-Lin/PConv_in_tf/blob/master/U_net/u_net_Struct.png)<br>Replace convolution with local convolution<br>
## Mask generation
Unlike the original,I used opencv to generate a mask and set the invalid part input to zero. In order to ensure the irregularity of the mask, without filling, the number of units in the mask part is also random, but at least a total of 4 * 5 units, up to 12 * 5 units (these can be set).<br>
## Use
### Generate images and masks:
```
num_mask:  the number of generated masks [required]
min_units:  the lower limit of the number of occlusion units in the mask [required]
max_units:  the upper limit of the occlusion unit in the mask [required]
masks_path:  the storage path of the generated mask [optional], default=~/inpainting/masks/
original_images_path:  the original image path [optional], default=~/inpainting/original-images/
masked_images_path:  The mask acts on the path after the original image,this is the training sample we generated. [optional], default=~/inpainting/masked-images/
```
Example of use:
```
PATH_TO_MASK=path/to/mask
PATH_TO_IMAGES=path/to/images
PATH_TO_MASKED_IMAGES=path/to/mased/images

python generate_image_mask.py \
    --num_mask=6 \
    --min_units=5 \
    --max_units=12 \
    --masks_path=$PATH_TO_MASK \
    --original_images_path=$PATH_TO_IMAGES \
    --masked_images_path=$PATH_TO_MASKED_IMAGES
```
### Training:
```
im_path:  generated training image path
vgg_path:  path to vgg.mat
num_epoch:  number of iterations
logdir: the path of the graph
save_path:  model save path
```

Example of use:

```
python train.py \
    --im_path="D:/inpaint/imfilenew/" \
    --vgg_path="D:/inpaint/imagenet-vgg-verydeep-19.mat" \
    --num_epoch=5 \
    --logdir="D:/inpaint/Logdir/" \
    --save_path="D:/inpaint/ckptdir/"
```

### Some instructions:
* At present, only the batch size is 1 and any batch number will be implemented later.
* During the operation, it was found that the result of the loss factor of the original text was too large to affect the network optimization, so the input image was mapped between [-1, 1] (as many seniors did), intuitively It can take into account visual effects and network optimization; for the original vgg structure definition file, the preprocessing part of the image is changed from: MEAN_PIXEL = (123.68,116.779,103.939]) to MEAN_PIXEL = (123.68/127.5-1,116.779/127.5-1,103.939/127.5-1]), but at present the method has not produced definitive results, I sincerely hope to communicate with you!
* [Download vgg](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)<br>
### Acknowledgement:
[fast-style-transfer](https://github.com/lengstrom/fast-style-transfer)
