import requests
import pandas as pd

base_URL = 'https://data.yt8m.org/2/j/i/'
youtube_url = 'https://www.youtube.com/watch?v='


def getURL(vid_id):
    URL = base_URL + vid_id[:-2] + '/' + vid_id + '.js'
    response = requests.get(URL, verify = False)
    if response.status_code == 200:
        return youtube_url + response.text[10:-3]


def getVideoInfo(vid_id, video_tags_path, top_k):
    video_url = getURL(vid_id)

    entire_video_tags = pd.read_csv(video_tags_path)
    video_tags_info = entire_video_tags.loc[entire_video_tags["vid_id"] == vid_id]
    video_tags = []
    for i in range(1, top_k + 1):
        video_tag_tuple = video_tags_info["segment" + str(i)].values[0]     # ex: "mobile-phone:0.361"
        video_tags.append(video_tag_tuple.split(":")[0])

    return {
        "video_url": video_url,
        "video_tags": video_tags
    }
