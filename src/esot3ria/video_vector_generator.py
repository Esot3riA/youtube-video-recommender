import pandas as pd
import numpy as np
from gensim.models import Word2Vec

BATCH_SIZE = 1000


def vectorization_video():
    print('[0.1 0.2]')


if __name__ == '__main__':
    tag_vectors = Word2Vec.load("tag_vectors.model").wv
    video_vectors = Word2Vec().wv   # Empty model

    # Load video recommendation tags.
    video_tags = pd.read_csv('kaggle_solution_40k.csv')

    # Define batch variables.
    batch_video_ids = []
    batch_video_vectors = []
    error_tags = []

    for i, row in video_tags.iterrows():
        video_id = row[0]
        video_vector = np.zeros(100)
        for segment_index in range(1, 6):
            tag, weight = row[segment_index].split(":")
            if tag in tag_vectors.vocab:
                video_vector = video_vector + (tag_vectors[tag] * float(weight))
            else:
                # Pass if tag is unknown
                if tag not in error_tags:
                    error_tags.append(tag)

        batch_video_ids.append(video_id)
        batch_video_vectors.append(video_vector)
        # Add video vectors.
        if (i+1) % BATCH_SIZE == 0:
            video_vectors.add(batch_video_ids, batch_video_vectors)
            batch_video_ids = []
            batch_video_vectors = []
            print("Video vectors created: ", i+1)

    # Add rest of video vectors.
    video_vectors.add(batch_video_ids, batch_video_vectors)
    print("error tags: ")
    print(error_tags)

    video_vectors.save("video_vectors.model")

    # Usage
    # video_vectors = Word2Vec().wv.load("video_vectors.model")
    # video_vectors.most_similar("XwFj", topn=5)
