# -*- coding: utf-8 -*-

# Sample Python code for youtube.commentThreads.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os
import googleapiclient.discovery


def main():

    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyD9kVeSJ1Vl_p454l0Maqk15wG6_LWwcPc"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part='snippet,replies',
        maxResults=20,
        videoId='J0Raq0F76ZU'
    )
    response = request.execute()

    request_2 = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id="J0Raq0F76ZU"
    )
    response_2 = request_2.execute()

    # temp_video_id = 'J0Raq0F76ZU'
    # print(response)
    # print(response.keys())
    # print(response['pageInfo'])
    # print(response['nextPageToken'])
    # print(response['items'][0]['snippet']['topLevelComment'])

    print(response_2['items'][0]['snippet']['title'])
    print(response_2['items'][0]['snippet']['description'])


    return 0


def html_template(display_name=None, text=None, profile_image=None, publish_time=None):
    if (display_name is None) or (text is None) or (profile_image is None) or (publish_time is None):
        return ''
    html_code = '<div class="media"><div class="pull-left media-top"><img src="' + profile_image + '" class="media-object rounded-circle" style="width:40px" alt=""></div><div class="media-body"><h6 class="media-heading">' + display_name + '<span style="font-size:0.65em;color:#808080"> ' + publish_time + '</span></h6><p>' + text + '</p></div></div>'
    return html_code


def comments_process(snippet=None):
    if snippet is None:
        return ''

    html_code = ''
    item_size = snippet['pageInfo']['totalResults']
    for comment_count in range(0, item_size):
        comment_body = snippet['items'][comment_count]['snippet']['topLevelComment']
        display_name = comment_body['snippet']['authorDisplayName']
        text = comment_body['snippet']['textDisplay']
        profile_image = comment_body['snippet']['authorProfileImageUrl']
        publish_time = '   ' + comment_body['snippet']['updatedAt']
        publish_time= publish_time.replace("T", " ")
        publish_time = publish_time.replace("Z", "")

        comments_code = html_template(display_name=display_name, text=text, profile_image=profile_image, publish_time=publish_time)
        html_code = html_code + comments_code
    return html_code


def get_comments(video_id=None, next_token=None):
    if video_id is None:
        return ''

    # html_file = '<div class="media"><div class="pull-left media-top"><img src="https://yt3.ggpht.com/a/AATXAJyA8wYBzlUTUpO3wemWSN9kWKIg6jkFRxQadRAQTw=s48-c-k-c0xffffffff-no-rj-mo" class="media-object" style="width:60px" alt=""></div><div class="media-body"><h5 class="media-heading">John Doe <span style="font-size:0.65em;color:#808080">This whole sentence is in small letters.</span></h5><p>这是一些示例文本..</p></div></div>'

    # Pull the comments from YouTube

    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyD9kVeSJ1Vl_p454l0Maqk15wG6_LWwcPc"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part='snippet,replies',
        maxResults=20,
        videoId=video_id
    )
    response = request.execute()
    html_file = comments_process(response)

    return html_file


def get_video_info(video_id=None):
    if video_id is None:
        return ''
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = "AIzaSyD9kVeSJ1Vl_p454l0Maqk15wG6_LWwcPc"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()

    v_title = response['items'][0]['snippet']['title']
    v_description = response['items'][0]['snippet']['description']

    return v_title, v_description


if __name__ == "__main__":
    main()