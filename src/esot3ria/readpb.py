import tensorflow as tf
import numpy as np

frame_lvl_record = "test0000.tfrecord"

feat_rgb = []
feat_audio = []

for example in tf.python_io.tf_record_iterator(frame_lvl_record):
    tf_seq_example = tf.train.SequenceExample.FromString(example)
    test = tf_seq_example.SerializeToString()
    n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)
    sess = tf.InteractiveSession()
    rgb_frame = []
    audio_frame = []
    # iterate through frames
    for i in range(n_frames):
        rgb_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['rgb']
                .feature[i].bytes_list.value[0], tf.uint8)
            , tf.float32).eval())
        audio_frame.append(tf.cast(tf.decode_raw(
            tf_seq_example.feature_lists.feature_list['audio']
                .feature[i].bytes_list.value[0], tf.uint8)
            , tf.float32).eval())

    sess.close()

    feat_audio.append(audio_frame)
    feat_rgb.append(rgb_frame)
    break

print('The first video has %d frames' %len(feat_rgb[0]))