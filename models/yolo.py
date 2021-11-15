import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D
from tensorflow import keras

from absl.flags import FLAGS
from absl import app, flags
from pathlib import Path
import logging
import yaml
import math
import numpy as np
LOGGER = logging.getLogger(__name__)






class Conv2d(keras.layers.Layer):
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        super(Conv2d, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(
            c2, k, s, 'VALID', use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None, )

    def call(self, inputs):
        return self.conv(inputs)


class LeakyRelu(object):
    def __call__(self, x):
        return tf.nn.leaky_relu(x)

class Conv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='SAME', groups=1):
        super(Conv, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, padding, groups=groups, use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))
        self.bn = BatchNormalization()
        self.activation = LeakyRelu()

    def call(self, x):
        return self.activation(self.bn(self.conv(x)))



# class DWConv(Layer):
#     def __init__(self, filters, kernel_size, strides):
#         super(DWConv, self).__init__()
#         self.conv = Conv(filters, kernel_size, strides, groups=1)  # Todo
#
#     def call(self, x):
#         return self.conv(x)


class Focus(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='SAME'):
        super(Focus, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, padding)

    def call(self, x):
        return self.conv(tf.concat([x[..., ::2, ::2, :],
                                    x[..., 1::2, ::2, :],
                                    x[..., ::2, 1::2, :],
                                    x[..., 1::2, 1::2, :]],
                                   axis=-1))



class Bottleneck(Layer):
    def __init__(self, units, shortcut=True, expansion=0.5):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv(int(units * expansion), 1, 1)
        self.conv2 = Conv(units, 3, 1)
        self.shortcut = shortcut

    def call(self, x):
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))


class BottleneckCSP(Layer):
    def __init__(self, units, n_layer=1, shortcut=True, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        units_e = int(units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(units, 1, 1)
        self.bn = BatchNormalization(momentum=0.03)
        self.activation = LeakyRelu()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))


class C3(Layer):
    def __init__(self, units, n_layer=1, shortcut=False, expansion=0.5):
        super(C3, self).__init__()
        units_e = int(units)  # hidden channels
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units, 1, 1)
        self.bn = BatchNormalization()
        self.activation = LeakyRelu()
        self.modules = tf.keras.Sequential([Bottleneck(units_e, shortcut, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        x1 = self.conv1(x)
        y1 = self.modules(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.activation(self.bn(tf.concat([y1, y2], axis=-1))))


class SPP(Layer):
    def __init__(self, units, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        units_e = units // 2  # Todo:
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv(units, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in kernels]  # Todo: padding check

    def call(self, x):
        x = self.conv1(x)
        return self.conv2(tf.concat([x] + [module(x) for module in self.modules], axis=-1))


class SPPCSP(Layer):
    # Cross Stage Partial Networks
    def __init__(self, units, n=1, shortcut=False, expansion=0.5, kernels=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        units_e = int(2 * units * expansion)
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv2D(units_e, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv(units_e, 3, 1)
        self.conv4 = Conv(units_e, 1, 1)
        self.modules = [MaxPool2D(pool_size=x, strides=1, padding='same') for x in kernels]
        self.conv5 = Conv(units_e, 1, 1)
        self.conv6 = Conv(units_e, 3, 1)
        self.bn = BatchNormalization()
        self.act = LeakyRelu()
        self.conv7 = Conv(units, 1, 1)

    def call(self, x):
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(tf.concat([x1] + [module(x1) for module in self.modules], axis=-1)))
        y2 = self.conv2(x)
        return self.conv7(self.act(self.bn(tf.concat([y1, y2], axis=-1))))


class Upsample(Layer):
    def __init__(self, i=None, ratio=2, method='bilinear'):
        super(Upsample, self).__init__()
        self.ratio = ratio
        self.method = method

    def call(self, x):
        return tf.image.resize(x, (tf.shape(x)[1] * self.ratio, tf.shape(x)[2] * self.ratio), method=self.method)


class Concat(Layer):
    def __init__(self, dims=-1):
        super(Concat, self).__init__()
        self.dims = dims

    def call(self, x):
        return tf.concat(x, self.dims)


def parse_model(yaml_dict):  # model_dict, input_channels(3)
        anchors, nc = yaml_dict['anchors'], yaml_dict['nc']
        depth_multiple, width_multiple = yaml_dict['depth_multiple'], yaml_dict['width_multiple']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (nc + 5)
        layers = []
        # from, number, module, args
        for i, (f, number, module, args) in enumerate(yaml_dict['backbone'] + yaml_dict['head']):
            # all component is a Class, initialize here, call in self.forward
            module = eval(module) if isinstance(module, str) else module
            for j, arg in enumerate(args):
                try:
                    args[j] = eval(arg) if isinstance(arg, str) else arg  # eval strings, like Detect(nc, anchors)
                except:
                    pass
            number = max(round(number * depth_multiple), 1) if number > 1 else number  # control the model scale
            if module in [Conv2D, Conv, Bottleneck, SPP, Focus, BottleneckCSP, C3]:
                c2 = args[0]
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]
                if module in [BottleneckCSP, C3, SPPCSP]:
                    args.insert(1, number)
                    number = 1
            modules = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)    
            modules.i, modules.f = i, f
            layers.append(modules)
        return layers

class Detect(keras.layers.Layer):
    def __init__(self, nc=20, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer
        super(Detect, self).__init__()
        self.stride = np.array([8, 16, 32], np.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.cast(tf.reshape(anchors, [self.nl, -1, 2]), tf.float32)
        self.m = [Conv2D(self.no * self.na, 1, use_bias=False) for _ in range(self.nl)]
        self.training = False  # set to False after building model
        self.imgsz = imgsz

    def call(self, inputs):
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            _, grid1, grid2, _ = self.m[i](inputs[i]).shape
  #          x[i] = tf.transpose(tf.reshape(x[i], [-1, ny * nx, self.na, self.no]), [0, 2, 1, 3])
            x[i] = tf.reshape(self.m[i](inputs[i]), (-1, grid1, grid2, self.na, self.no))          
            if not self.training:  # inference
                y = tf.sigmoid(x[i])
                
                grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x)
                grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2),tf.float32)  
                xy, wh, conf, classes = tf.split(y, (2, 2, 1, self.nc), axis=-1)
                pred_xy = (xy * 2. - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
                pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]
            
                out = tf.concat([pred_xy, pred_wh, conf, classes], axis=-1)
                z.append(out)
#                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
  #              wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
#                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
 #               wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
 #               y = tf.concat([xy, wh, y[..., 4:]], -1)
 #               z.append(tf.reshape(y, [-1, 3 * ny * nx, self.no]))

        return z

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)

class Model(object):
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=20, model=None, imgsz=(640, 640)):  # model, channels, classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.imgsz =imgsz
        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model = parse_model(self.yaml)
        if isinstance(model, Detect):
            # transfer the anchors to grid coordinator, 3 * 3 * 2
            model.anchors /= tf.reshape(module.stride, [-1, 1, 1])
            
    def __call__(self, img_size, name='yolo'):
        x = tf.keras.Input([img_size, img_size, 3])
        output = self.forward(x)
        return tf.keras.Model(inputs=x, outputs=output, name=name)
        
    def forward(self, inputs, tf_nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45,
                conf_thres=0.25):
        y = []  # outputs
        x = inputs
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
#                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x)

        return x

    # @staticmethod
    # def _xywh2xyxy(xywh):
    #     # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #     x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
    #     return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)
        
        
class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        # wrap map_fn to avoid TypeSpec related error https://stackoverflow.com/a/65809989/3036450
        return tf.map_fn(self._nms, input,
                         fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
                         name='agnostic_nms')

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # agnostic NMS
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(
            boxes, scores_inp, max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres)
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(selected_boxes,
                              paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
                              mode="CONSTANT", constant_values=0.0)
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(selected_scores,
                               paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                               mode="CONSTANT", constant_values=-1.0)
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(selected_classes,
                                paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
                                mode="CONSTANT", constant_values=-1.0)
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections

class Loss(object):
      def __init__(self, anchors, iou_thres, num_classes=20, img_size=640, label_smoothing=0):
        self.anchors = anchors
        self.strides = [8, 16, 32]
        self.iou_thres = iou_thres
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce_conf = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.bce_class = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                                            label_smoothing=label_smoothing)
      
      def __call__(self, y_true, y_pred):
        iou_loss_all = obj_loss_all = class_loss_all = tf.zeros(1)
        balance = [4.0, 1.0, 0.4] if len(y_pred) == 3 else [4.0, 1.0, 0.25, 0.06]
        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            true_box, true_obj, true_class = tf.split(true, (4, 1, -1), axis=-1)
            pred_box, pred_obj, pred_class = tf.split(pred, (4, 1, -1), axis=-1)
            if tf.shape(true_class)[-1] == 1 and self.num_classes > 1:
                true_class = tf.squeeze(tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=self.num_classes, axis=-1), -2) 

            # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
            box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / (self.img_size ** 2)
            obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
            background_mask = 1.0 - obj_mask
            conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)

            # giou loss
            iou = bbox_iou(pred_box, true_box, xyxy=False, giou=True)            
            iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3

            # confidence loss
            conf_loss = self.bce_conf(true_obj, pred_obj)
            conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3

            # class loss
            class_loss = obj_mask * self.bce_class(true_class, pred_class)

            iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3]))
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3]))
            class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3]))

            iou_loss_all += iou_loss * balance[i]
            obj_loss_all += conf_loss * balance[i]
            class_loss_all += class_loss * self.num_classes * balance[i]  # to balance the 3 loss

        return (iou_loss_all, obj_loss_all, class_loss_all)
        
def bbox_iou(bbox1, bbox2, xyxy=False, giou=False, diou=False, ciou=False, epsilon=1e-9):
    assert bbox1.shape == bbox2.shape
    # giou loss: https://arxiv.org/abs/1902.09630
    if xyxy:
        b1x1, b1y1, b1x2, b1y2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]
    else:  # xywh -> xyxy
        b1x1, b1x2 = bbox1[..., 0] - bbox1[..., 2] / 2, bbox1[..., 0] + bbox1[..., 2] / 2
        b1y1, b1y2 = bbox1[..., 1] - bbox1[..., 3] / 2, bbox1[..., 1] + bbox1[..., 3] / 2
        b2x1, b2x2 = bbox2[..., 0] - bbox2[..., 2] / 2, bbox2[..., 0] + bbox2[..., 2] / 2
        b2y1, b2y2 = bbox2[..., 1] - bbox2[..., 3] / 2, bbox2[..., 1] + bbox2[..., 3] / 2

    # intersection area
    inter = tf.maximum(tf.minimum(b1x2, b2x2) - tf.maximum(b1x1, b2x1), 0) * \
            tf.maximum(tf.minimum(b1y2, b2y2) - tf.maximum(b1y1, b2y1), 0)

    # union area
    w1, h1 = b1x2 - b1x1 + epsilon, b1y2 - b1y1 + epsilon
    w2, h2 = b2x2 - b2x1+ epsilon, b2y2 - b2y1 + epsilon
    union = w1 * h1 + w2 * h2 - inter + epsilon

    # iou
    iou = inter / union

    cw = tf.maximum(b1x2, b2x2) - tf.minimum(b1x1, b2x1)
    ch = tf.maximum(b1y2, b2y2) - tf.minimum(b1y1, b2y1)
    enclose_area = cw * ch + epsilon
    giou = iou - 1.0 * (enclose_area - union) / enclose_area
    return tf.clip_by_value(giou, -1, 1)
    
def main(imgsz=(640, 640),  # inference size h,w
        batch_size=1,  # batch size
        ):

    # TensorFlow model
    im = tf.zeros((batch_size, 640, 640, 3))  # BHWC image
    tf_model = Model(cfg='yolov5s.yaml', imgsz=imgsz)
    y = tf_model.predict(im)  # inference

if __name__=='__main__':
    app.run(main)