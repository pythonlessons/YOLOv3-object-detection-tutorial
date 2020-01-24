import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2


import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from collect_XLM_images import *

class YOLO(object):
    _defaults = {
        "model_path": 'logs/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/CSGO_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "text_size" : 4,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

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

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList

'''
from textwrap import dedent
from lxml import etree
import glob
def CreateXMLfile(image, ObjectsList):
    Folder_to_save = "XML_images"
    os.chdir(Folder_to_save)
    print("Number of .jpg files =", len(glob.glob("*.jpg")))
    

    annotation = etree.Element("annotation")

    folder = etree.Element("folder")
    folder.text = os.path.basename(os.getcwd())
    annotation.append(folder)

    filename_xml = etree.Element("filename")
    filename_str = image.split(".")[0]
    filename_xml.text = image#filename_str + ".jpg"
    annotation.append(filename_xml)

    path = etree.Element("path")
    path.text = os.path.join(os.getcwd(), filename_str + ".jpg")
    annotation.append(path)

    source = etree.Element("source")
    annotation.append(source)

    database = etree.Element("database")
    database.text = "Unknown"
    source.append(database)

    size = etree.Element("size")
    annotation.append(size)

    width = etree.Element("width")
    height = etree.Element("height")
    depth = etree.Element("depth")

    img = cv2.imread(filename_xml.text)

    #try:
    width.text = str(img.shape[1])
    #except AttributeError:
        #os.chdir("Label")
        #continue
    height.text = str(img.shape[0])
    depth.text = str(img.shape[2])

    size.append(width)
    size.append(height)
    size.append(depth)

    segmented = etree.Element("segmented")
    segmented.text = "0"
    annotation.append(segmented)

    for Object in ObjectsList:
        class_name = Object[6]
        xmin_l = str(int(float(Object[1])))
        ymin_l = str(int(float(Object[0])))
        xmax_l = str(int(float(Object[3])))
        ymax_l = str(int(float(Object[2])))

        obj = etree.Element("object")
        annotation.append(obj)

        name = etree.Element("name")
        name.text = class_name
        obj.append(name)

        pose = etree.Element("pose")
        pose.text = "Unspecified"
        obj.append(pose)

        truncated = etree.Element("truncated")
        truncated.text = "0"
        obj.append(truncated)

        difficult = etree.Element("difficult")
        difficult.text = "0"
        obj.append(difficult)

        bndbox = etree.Element("bndbox")
        obj.append(bndbox)

        xmin = etree.Element("xmin")
        xmin.text = xmin_l
        bndbox.append(xmin)

        ymin = etree.Element("ymin")
        ymin.text = ymin_l
        bndbox.append(ymin)

        xmax = etree.Element("xmax")
        xmax.text = xmax_l
        bndbox.append(xmax)

        ymax = etree.Element("ymax")
        ymax.text = ymax_l
        bndbox.append(ymax)

    # write xml to file
    s = etree.tostring(annotation, pretty_print=True)
    with open(filename_str + ".xml", 'wb') as f:
        f.write(s)
        f.close()

    os.chdir("..")
'''
    
if __name__=="__main__":
    yolo = YOLO()
    image = 'home.jpg'
    r_image, ObjectsList = yolo.detect_img(image)
    #print(ObjectsList)
    if len(ObjectsList) > 0:
        CreateXMLfile(cv2.imread(image), ObjectsList)
    cv2.imshow(image, r_image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
    cv2.imwrite('home_detection.jpg', r_image)

    yolo.close_session()
