from My_Log import Log_Manager
import tensorflow as tf
import inception_preprocessing
import functools
from tensorflow.python.ops import control_flow_ops

#-------------------------------------------
def base_norm(single_img):
    Log_Manager.print_info("Base_norm")
    with tf.variable_scope("Base_norm"):
        single_img = single_img / 255.
        single_img = single_img - 0.5
        return single_img
#-------------------------------------------
def slice_and_resize(single_img):
    with tf.variable_scope('slice_and_resize'):
        height = single_img.get_shape().dims[0].value
        width = single_img.get_shape().dims[1].value
        # Random crop cut from the street sign image, resized to the same size.
        # Assures that the crop is covers at least 0.8 area of the input image.
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(single_img),
            bounding_boxes=tf.zeros([0, 0, 4]),
            min_object_covered=0.8,
            aspect_ratio_range=[0.8, 1.2],
            area_range=[0.8, 1.0],
            use_image_if_no_bounding_boxes=True)
        distorted_image = tf.slice(single_img, bbox_begin, bbox_size)

        # Randomly chooses one of the 4 interpolation methods
        distorted_image = random_selector(
            distorted_image,
            lambda x, method: tf.image.resize_images(x, [height, width], method),
            num_cases=4)
        distorted_image.set_shape([height, width, 3])
        return distorted_image
#-------------------------------------------
def distorting_color(single_image):
    with tf.variable_scope('distorting_color'):
        distorted_image = random_selector(single_image,
                                          functools.partial(distorting_color_pool,fast_mode=False,scope=None),
                                          num_cases=4)
        distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)
        return distorted_image

def random_selector(input,func,num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([func(control_flow_ops.switch(input, tf.equal(sel, case))[1], case)
                                      for case in range(num_cases)
                                      ])[0]

def distorting_color_pool(image,color_ordering=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return image
#-------------------------------------------

