from styx_msgs.msg import TrafficLight
import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import tensorflow as tf
import os
from PIL import Image
import cv2


class TLClassifier(object):
    _defaults_sim = {
        "model_path": 'light_classification/model_data_sim/yolo.h5',
        "anchors_path": 'light_classification/model_data_sim/yolo_anchors.txt',
        "classes_path": 'light_classification/model_data_sim/voc_classes.txt',
        "score": 0.6,
        "iou": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    _defaults_real = {
        "model_path": 'light_classification/model_data_real/yolo.h5',
        "anchors_path": 'light_classification/model_data_real/yolo_anchors.txt',
        "classes_path": 'light_classification/model_data_real/voc_classes.txt',
        "score": 0.6,
        "iou": 0.5,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    def __init__(self, is_sim, **kwargs):
        if is_sim:
            self.__dict__.update(self._defaults_sim)
        else:
            self.__dict__.update(self._defaults_real)
        self.__dict__.update(kwargs)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

        self.class_msgs = {'red': TrafficLight.RED,
                           'yellow': TrafficLight.YELLOW,
                           'green': TrafficLight.GREEN,
                           'unknown': TrafficLight.UNKNOWN}
        self.graph = tf.get_default_graph()

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def get_classification(self, image):

        start = timer()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        with self.graph.as_default():
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            score = out_scores[i]

            # print('state predicted : ', predicted_class)
            return self.class_msgs[predicted_class], score
            # return predicted_class, score
        end = timer()
        # print(end - start)
        return TrafficLight.UNKNOWN, -1.0

# if __name__ == "__main__":
#     t = TLClassifier()
#     img = cv2.imread('E:/Udacity-Capstone-Datasets/dataset-sdcnd-capstone/data/sim_training_data/sim_data_capture/left0003.jpg')
#     a, b = t.get_classification(img)
#     print(a,b)
