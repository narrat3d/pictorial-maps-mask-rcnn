'''
helper class for inference

code based on object_detection_tutorial.ipynb
'''
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection.utils import ops as utils_ops

def initialise_sessions(inference_model_path):
    output_tensor_names = [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():     
        # read in byte-string image like in train/eval   
        image_placeholder = tf.placeholder(tf.string)
        decoded_image = tf.image.decode_image(image_placeholder, channels=3, name="decoded_image")
        decoded_image.set_shape([None, None, 3])
        new_image_tensor = tf.expand_dims(decoded_image, 0)
        
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            
            tf.import_graph_def(od_graph_def, input_map={"image_tensor:0": new_image_tensor}, name='')
            
            # get rid of 'image_tensor' which would be disconnected from the main graph otherwise 
            # and expected in the feed_dict for session.run
            od_graph_def = tf.graph_util.extract_sub_graph(od_graph_def, output_tensor_names)

    detection_session = tf.Session(graph = detection_graph)
    
    ops = detection_graph.get_operations()        
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    
    tensor_dict = {}
    tensor_dict["image"] = decoded_image
    
    for output_tensor_name in output_tensor_names:
        tensor_name = output_tensor_name + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[output_tensor_name] = detection_graph.get_tensor_by_name(
                    tensor_name)
          
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    
    image_shape = tf.shape(new_image_tensor)
    # resizes the masks based on input image size
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image_shape[1], image_shape[2])
    detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    
    return (image_placeholder, detection_session, tensor_dict)


def infer(image_file_path, image_placeholder, detection_session, tensor_dict):
    with tf.gfile.GFile(image_file_path, 'rb') as fid:
        image_byte_string = fid.read()
    
    output_dict = detection_session.run(tensor_dict, feed_dict={image_placeholder: image_byte_string})

    image = Image.fromarray(output_dict["image"])
    image_width, image_height = image.size  

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    output_dict['detection_masks'] = output_dict['detection_masks'][0]

    detection_bounding_boxes = []
    detection_masks = []
    detection_scores = []
    detection_classes = []

    for i in range(output_dict['num_detections']):
        score = output_dict['detection_scores'][i]
        class_ = output_dict['detection_classes'][i]
        
        box = output_dict['detection_boxes'][i]
        
        box_min_y = int(image_height * box[0])
        box_min_x = int(image_width * box[1])
        box_max_y = int(image_height * box[2]) + 1
        box_max_x = int(image_width * box[3]) + 1
        
        mask = output_dict['detection_masks'][i]
        image_mask = Image.fromarray(mask * 255, "L")
            
        detection_bounding_boxes.append([box_min_x, box_min_y, box_max_x, box_max_y])
        detection_masks.append(image_mask)
        detection_scores.append(score.item())
        detection_classes.append(class_.item())
    
    return (detection_bounding_boxes, detection_masks, detection_scores, detection_classes, image)