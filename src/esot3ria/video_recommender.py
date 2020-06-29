from gensim.models import Word2Vec
import numpy as np

def recommend_videos(tags, tag_model_path, video_model_path, top_k):
    tag_vectors = Word2Vec.load(tag_model_path).wv
    video_vectors = Word2Vec().wv.load(video_model_path)
    error_tags = []

    video_vector = np.zeros(100)
    for (tag, weight) in tags:
        if tag in tag_vectors.vocab:
            video_vector = video_vector + (tag_vectors[tag] * float(weight))
        else:
            # Pass if tag is unknown
            if tag not in error_tags:
                error_tags.append(tag)

    similar_ids = [x[0] for x in video_vectors.similar_by_vector(video_vector, top_k)]
    return similar_ids
