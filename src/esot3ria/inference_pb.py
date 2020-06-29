import numpy as np
import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
import operator
import esot3ria.pb_util as pbutil
import esot3ria.video_recommender as recommender
import esot3ria.video_util as videoutil

# Define file paths.
MODEL_PATH = "./model/inference_model/segment_inference_model"
VOCAB_PATH = "../vocabulary.csv"
VIDEO_TAGS_PATH = "./kaggle_solution_40k.csv"
TAG_VECTOR_MODEL_PATH = "./tag_vectors.model"
VIDEO_VECTOR_MODEL_PATH = "./video_vectors.model"
SEGMENT_LABEL_PATH = "../segment_label_ids.csv"

# Define parameters.
TAG_TOP_K = 5
VIDEO_TOP_K = 10


def get_segments(batch_video_mtx, batch_num_frames, segment_size):
    """Get segment-level inputs from frame-level features."""
    video_batch_size = batch_video_mtx.shape[0]
    max_frame = batch_video_mtx.shape[1]
    feature_dim = batch_video_mtx.shape[-1]
    padded_segment_sizes = (batch_num_frames + segment_size - 1) // segment_size
    padded_segment_sizes *= segment_size
    segment_mask = (
            0 < (padded_segment_sizes[:, np.newaxis] - np.arange(0, max_frame)))

    # Segment bags.
    frame_bags = batch_video_mtx.reshape((-1, feature_dim))
    segment_frames = frame_bags[segment_mask.reshape(-1)].reshape(
        (-1, segment_size, feature_dim))

    # Segment num frames.
    segment_start_times = np.arange(0, max_frame, segment_size)
    num_segments = batch_num_frames[:, np.newaxis] - segment_start_times
    num_segment_bags = num_segments.reshape((-1))
    valid_segment_mask = num_segment_bags > 0
    segment_num_frames = num_segment_bags[valid_segment_mask]
    segment_num_frames[segment_num_frames > segment_size] = segment_size

    max_segment_num = (max_frame + segment_size - 1) // segment_size
    video_idxs = np.tile(
        np.arange(0, video_batch_size)[:, np.newaxis], [1, max_segment_num])
    segment_idxs = np.tile(segment_start_times, [video_batch_size, 1])
    idx_bags = np.stack([video_idxs, segment_idxs], axis=-1).reshape((-1, 2))
    video_segment_ids = idx_bags[valid_segment_mask]

    return {
        "video_batch": segment_frames,
        "num_frames_batch": segment_num_frames,
        "video_segment_ids": video_segment_ids
    }


def format_predictions(video_ids, predictions, top_k, whitelisted_cls_mask=None):
    batch_size = len(video_ids)
    for video_index in range(batch_size):
        video_prediction = predictions[video_index]
        if whitelisted_cls_mask is not None:
            # Whitelist classes.
            video_prediction *= whitelisted_cls_mask
        top_indices = np.argpartition(video_prediction, -top_k)[-top_k:]
        line = [(class_index, predictions[video_index][class_index])
                for class_index in top_indices]
        line = sorted(line, key=lambda p: -p[1])
        yield (video_ids[video_index] + "," +
               " ".join("%i %g" % (label, score) for (label, score) in line) +
               "\n").encode("utf8")


def normalize_tag(tag):
    if isinstance(tag, str):
        new_tag = tag.lower().replace('[^a-zA-Z]', ' ')
        if new_tag.find(" (") != -1:
            new_tag = new_tag[:new_tag.find(" (")]
        new_tag = new_tag.replace(" ", "-")
        return new_tag
    else:
        return tag


def inference_pb(file_path, threshold):
    VIDEO_TOP_K = int(threshold)
    inference_result = {}
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # 0. Import SequenceExample type target from pb.
        target_video = pbutil.convert_pb(file_path)

        # 1. Load video features from pb.
        video_id_batch_val = np.array([b'video'])
        n_frames = len(target_video.feature_lists.feature_list['rgb'].feature)
        # Restrict frame size to 300
        if n_frames > 300:
            n_frames = 300
        video_batch_val = np.zeros((300, 1152))
        for i in range(n_frames):
            video_batch_rgb_raw = target_video.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0]
            video_batch_rgb = np.array(tf.cast(tf.decode_raw(video_batch_rgb_raw, tf.float32), tf.float32).eval())
            video_batch_audio_raw = target_video.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0]
            video_batch_audio = np.array(tf.cast(tf.decode_raw(video_batch_audio_raw, tf.float32), tf.float32).eval())
            video_batch_val[i] = np.concatenate([video_batch_rgb, video_batch_audio], axis=0)
        video_batch_val = np.array([video_batch_val])
        num_frames_batch_val = np.array([n_frames])

        # Restore checkpoint and meta-graph file.
        if not gfile.Exists(MODEL_PATH + ".meta"):
            raise IOError("Cannot find %s. Did you run eval.py?" % MODEL_PATH)
        meta_graph_location = MODEL_PATH + ".meta"
        logging.info("loading meta-graph: " + meta_graph_location)

        with tf.device("/cpu:0"):
            saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
        logging.info("restoring variables from " + MODEL_PATH)
        saver.restore(sess, MODEL_PATH)
        input_tensor = tf.get_collection("input_batch_raw")[0]
        num_frames_tensor = tf.get_collection("num_frames")[0]
        predictions_tensor = tf.get_collection("predictions")[0]

        # Workaround for num_epochs issue.
        def set_up_init_ops(variables):
            init_op_list = []
            for variable in list(variables):
                if "train_input" in variable.name:
                    init_op_list.append(tf.assign(variable, 1))
                    variables.remove(variable)
            init_op_list.append(tf.variables_initializer(variables))
            return init_op_list

        sess.run(
            set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))

        whitelisted_cls_mask = np.zeros((predictions_tensor.get_shape()[-1],),
                                        dtype=np.float32)
        with tf.io.gfile.GFile(SEGMENT_LABEL_PATH) as fobj:
            for line in fobj:
                try:
                    cls_id = int(line)
                    whitelisted_cls_mask[cls_id] = 1.
                except ValueError:
                    # Simply skip the non-integer line.
                    continue

        # 2. Make segment features.
        results = get_segments(video_batch_val, num_frames_batch_val, 5)
        video_segment_ids = results["video_segment_ids"]
        video_id_batch_val = video_id_batch_val[video_segment_ids[:, 0]]
        video_id_batch_val = np.array([
            "%s:%d" % (x.decode("utf8"), y)
            for x, y in zip(video_id_batch_val, video_segment_ids[:, 1])
        ])
        video_batch_val = results["video_batch"]
        num_frames_batch_val = results["num_frames_batch"]
        if input_tensor.get_shape()[1] != video_batch_val.shape[1]:
            raise ValueError("max_frames mismatch. Please re-run the eval.py "
                             "with correct segment_labels settings.")

        predictions_val, = sess.run([predictions_tensor],
                                    feed_dict={
                                        input_tensor: video_batch_val,
                                        num_frames_tensor: num_frames_batch_val
                                    })

        # 3. Make vocabularies.
        voca_dict = {}
        vocabs = open(VOCAB_PATH, 'r')
        while True:
            line = vocabs.readline()
            if not line: break
            vocab_dict_item = line.split(",")
            if vocab_dict_item[0] != "Index":
                voca_dict[vocab_dict_item[0]] = vocab_dict_item[3]
        vocabs.close()

        # 4. Make combined scores.
        combined_scores = {}
        for line in format_predictions(video_id_batch_val, predictions_val, TAG_TOP_K, whitelisted_cls_mask):
            segment_id, preds = line.decode("utf8").split(",")
            preds = preds.split(" ")
            pred_cls_ids = [int(preds[idx]) for idx in range(0, len(preds), 2)]
            pred_cls_scores = [float(preds[idx]) for idx in range(1, len(preds), 2)]
            for i in range(len(pred_cls_ids)):
                if pred_cls_ids[i] in combined_scores:
                    combined_scores[pred_cls_ids[i]] += pred_cls_scores[i]
                else:
                    combined_scores[pred_cls_ids[i]] = pred_cls_scores[i]

        combined_scores = sorted(combined_scores.items(), key=operator.itemgetter(1), reverse=True)
        demoninator = float(combined_scores[0][1] + combined_scores[1][1]
                            + combined_scores[2][1] + combined_scores[3][1] + combined_scores[4][1])

        tag_result = []
        for itemIndex in range(TAG_TOP_K):
            segment_tag = str(voca_dict[str(combined_scores[itemIndex][0])])
            normalized_tag = normalize_tag(segment_tag)
            tag_percentage = format(combined_scores[itemIndex][1] / demoninator, ".3f")
            tag_result.append((normalized_tag, tag_percentage))

        # 5. Create recommend videos info, Combine results.
        recommend_video_ids = recommender.recommend_videos(tag_result, TAG_VECTOR_MODEL_PATH,
                                                           VIDEO_VECTOR_MODEL_PATH, VIDEO_TOP_K)
        video_result = [videoutil.getVideoInfo(ids, VIDEO_TAGS_PATH, TAG_TOP_K) for ids in recommend_video_ids]

        inference_result = {
            "tag_result": tag_result,
            "video_result": video_result
        }

        # 6. Dispose instances.
        sess.close()

    return inference_result


if __name__ == '__main__':
    filepath = "../featuremaps/features(beauty).pb"
    result = inference_pb(filepath, 5)
    print(result)
