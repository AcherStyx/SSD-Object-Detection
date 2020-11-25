# COCO数据集导入

构建适应Tensorflow的`tf.data.Dataset`类，该数据集以`(图像, 标签)`的格式返回COCO数据集中的图像的标签。

迭代返回值：

- 图像，使用三维的numpy.ndarray储存的图像。

- 标签，category_id（使用数字索引表示的类别，由COCO在annotation中指定了默认值），x，y，w，h

使用时请将完整的coco数据集下载下来，以如下的目录形式储存。

```
>tree ./coco
...
E:...\COCO
├─annotations
├─train2017
└─val2017
```

`COCODataLoaderConfig`仅接收以上存放COCO数据集的文件夹的路径作为参数。

***

Google translate version:

# Load COCO data set

Construct the `tf.data.Dataset` class adapted to Tensorflow, which returns the tags of the images in the COCO dataset in the format of `(image, label)`.

Iteration return value:

 - Image: the image stored in a three-dimensional numpy.ndarray.

 - Label: category_id (the category represented by a numeric index, the default value is specified by COCO in the annotation file), x, y, w, h

Please download the complete coco data set when using it and store it in the following directory.

```
>tree ./coco
...
E:...\COCO
├─annotations
├─train2017
└─val2017
```

`COCODataLoaderConfig` only accepts the path of the folder where the COCO data set is stored as a parameter.