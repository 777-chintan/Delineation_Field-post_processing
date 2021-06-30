import tensorflow as tf
import cv2
import numpy as np
import os
import random
from IPython import embed

IMAGE_PATH = "/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/test/images/"

input_names = ['image_tensor']

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]),
                  (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale,
                (255, 255, 255), thickness)

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

detection_graph1 = tf.Graph()

with detection_graph1.as_default():
    trt_graph1 = get_frozen_graph('/home/chintu/Desktop/maskrcnn/maskrcnn_wth_tensorflow_object_detection_api/training_data/CP_23_03_2021/saved_model/frozen_inference_graph.pb')
    [tf_input, tf_scores, tf_boxes, tf_classes, tf_num_detections, 
    tf_masks] = tf.import_graph_def(trt_graph1, 
            return_elements=['image_tensor:0', 'detection_scores:0', 
            'detection_boxes:0', 'detection_classes:0','num_detections:0', 
            'detection_masks:0'])
    tf_sess1 = tf.Session(graph=detection_graph1)
    # model = tf.saved_model.loader.load(tf_sess1, ["serve"], "/Users/vedanshu/saved_model/tan_tomato_l1_mask_rcnn_inception_v2")
    
# tf_input = tf_sess1.graph.get_tensor_by_name(input_names[0] + ':0')
# tf_scores = tf_sess1.graph.get_tensor_by_name('detection_scores:0')
# tf_boxes = tf_sess1.graph.get_tensor_by_name('detection_boxes:0')
# tf_classes = tf_sess1.graph.get_tensor_by_name('detection_classes:0')
# tf_num_detections = tf_sess1.graph.get_tensor_by_name('num_detections:0')
# tf_masks = tf_sess1.graph.get_tensor_by_name('detection_masks:0')

for img in os.listdir(IMAGE_PATH):
    if img.endswith("jpg"):
        print("inferencing on ", img)
        image = os.path.join(IMAGE_PATH, img)
        # image = "/Users/vedanshu/test1.jpg"
        image = cv2.imread(image)
        _image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scores, boxes, classes, num_detections, masks = tf_sess1.run([tf_scores, tf_boxes, tf_classes, tf_num_detections, tf_masks], feed_dict={tf_input: _image[None, ...]})

        boxes = boxes[0]  # index by 0 to remove batch dimension
        scores = scores[0]
        classes = classes[0]
        num_detections = int(num_detections[0])
        masks = masks[0]

        # embed()
        boxes_pixels = []
        for i in range(num_detections):
            # scale box to image coordinates
            box = boxes[i] * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            box = np.round(box).astype(int)
            boxes_pixels.append(box)

        boxes_pixels = np.array(boxes_pixels)

        for i in range(num_detections):
            if scores[i] > 0.5:
                box = boxes_pixels[i]
                box = np.round(box).astype(int)
                (startY, startX, endY, endX) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY
                mask = masks[i]
                mask = cv2.resize(mask, (boxW, boxH))

                mask = (mask > 0.5)
                mask=(mask*255).astype(np.uint8)
                _bgd_mask = cv2.bitwise_not(mask)
                roi = image[startY:endY, startX:endX]
                _bgd = cv2.bitwise_and(roi,roi,mask = _bgd_mask)
                roi = cv2.bitwise_and(roi,roi,mask = mask)
                _roi = roi*[0,0,0.6]
                roi = _bgd + _roi
                image[startY:endY, startX:endX] = roi

                # roi = image[startY:endY, startX:endX]
                # roi = cv2.bitwise_and(roi,roi,mask = (mask*255).astype(np.uint8))
                # roi = image[startY:endY, startX:endX][mask]
                # color = random.choice(COLORS)
                # blended = (0.6 * roi).astype("uint8")
                # image[startY:endY, startX:endX][mask] = blended*[0,0,0.8]
  
                image = cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                label = "{}:{:.2f}".format(int(classes[i]), scores[i])
                draw_label(image, (box[1], box[0]), label)

                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("/home/chintu/Desktop/val_output/"+img, image)
        # cv2.waitKey(0)

# cv2.destroyAllWindows()


