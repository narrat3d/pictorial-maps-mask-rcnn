'''
helper class for inference
'''
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import ops as utils_ops
from character_segmentation import config


def initialise_sessions(inference_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_graph = tf.Graph()
    with image_graph.as_default():
        image_placeholder = tf.placeholder(tf.string)
        image_tensor = tf.image.decode_image(image_placeholder, channels=3)
        image_tensor.set_shape([None, None, 3])
        image_tensor = tf.expand_dims(image_tensor, 0)

    image_session = tf.Session(graph = image_graph)
    detection_session = tf.Session(graph = detection_graph)
    
    ops = detection_graph.get_operations()        
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    
    tensor_dict = {}
    for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(
                    tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, 1000, 1000)
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
    detection_image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    return (image_session, image_tensor, image_placeholder, detection_session, tensor_dict, detection_image_tensor)


def infer(image_session, image_tensor, image_placeholder, detection_session, 
           detection_tensor, detection_placeholder, image_file_path):   
    with tf.gfile.GFile(image_file_path, 'rb') as fid:
        image_byte_string = fid.read()
    
    decoded_image = image_session.run(image_tensor, feed_dict={image_placeholder: image_byte_string })
    
    image = Image.fromarray(decoded_image[0])
    image_width, image_height = image.size               
     
    output_dict = detection_session.run(detection_tensor, feed_dict={detection_placeholder: decoded_image })

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    output_dict['detection_masks'] = output_dict['detection_masks'][0]

    detection_bounding_boxes = []
    detection_masks = []

    for i in range(0, len(output_dict['detection_scores'])):
        score = output_dict['detection_scores'][i]
        
        if (score == 0):
            break
        
        box = output_dict['detection_boxes'][i]
        
        box_min_y = int(image_height * box[0])
        box_min_x = int(image_width * box[1])
        box_max_y = int(image_height * box[2]) + 1
        box_max_x = int(image_width * box[3]) + 1
        
        mask = output_dict['detection_masks'][i]
            
        if score > config.INFERENCE_THRESHOLD:
            detection_bounding_boxes.append([box_min_x, box_min_y, box_max_x, box_max_y])
            detection_masks.append(mask)
    
    return (detection_bounding_boxes, detection_masks, image)