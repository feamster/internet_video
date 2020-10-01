# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python

import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors

from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

import pickle

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]

video_list_templates = '<li class="list-group-item list-group-item-action flex-column align-items-start border-0"><div class="container"><div class="row"><div class="col"><a href="VIDEO_LINK_URL"><img src="VIDEO_THUMBNAIL_URL" alt="Barca"class=" img-fluid" width="100%"/></a></div><div class="col"><div class="d-flex w-100 justify-content-between" style="max-height:40px; overflow: hidden; text-overflow:ellipsis; hover:{overflow: visible;}"><a href="VIDEO_LINK_URL" style="color: black"> <h6 class="mb-1"> Video_Title </h6> </a></div><small> <a href="PUBLISHER_LINK_URL" target="_blank" rel="noopener noreferrer" style="color: black">Video_Publisher</a> </small><br><small> NUM_OF_VIEWS views &bull; PUBLISH_TIME</small></div></div></div></li>'


def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "client_secret_774470031846-9nraa6iu1jlkt7bsest4kbr0onud83e5.apps.googleusercontent.com.json"

    credentials = None
    if os.path.exists('oauth_key.pickle'):
        with open('oauth_key.pickle', 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes)
            credentials = flow.run_console()
        # Save the credentials for the next run
        with open('oauth_key.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

    request = youtube.search().list(
        part="snippet",
        maxResults=5,
        relatedToVideoId="jIdq3QMbHCI",
        type="video"
    )
    response = request.execute()

    print(response)

def process_video_info(snippet=None):
    if snippet is None:
        return ''
    html_codes = ''
    video_list = snippet['items']
    for idx in range(0, len(video_list)):
        video_title = video_list[idx]['snippet']['title']
        thumbnail_link = video_list[idx]['snippet']['thumbnails']['medium']['url']
        publisher_link = "https://www.youtube.com/channel/" + video_list[idx]['snippet']['channelId']
        publisher_name = video_list[idx]['snippet']['channelTitle']
        publish_time = video_list[idx]['snippet']['publishedAt']
        publish_time= publish_time.split("T")[0]
        video_link = 'https://www.youtube.com/watch?v=' + video_list[idx]['id']
        view_num = int(video_list[idx]['statistics']['viewCount'])
        view_num_text = str(view_num)
        if view_num > 1000000:
            view_num_text = str(format(view_num/1000000, '.1f')) + 'M'
        elif view_num > 1000:
            view_num_text = str(format(view_num/1000, '.1f')) + 'K'

        t_html = video_list_templates.replace("VIDEO_LINK_URL", video_link)
        t_html = t_html.replace("VIDEO_THUMBNAIL_URL", thumbnail_link)
        t_html = t_html.replace("PUBLISHER_LINK_URL", "publisher_link")
        t_html = t_html.replace("NUM_OF_VIEWS", view_num_text)
        t_html = t_html.replace("PUBLISH_TIME", publish_time)
        t_html = t_html.replace("Video_Publisher", publisher_name)
        t_html = t_html.replace("Video_Title", video_title)

        print('video_info: ', video_title, thumbnail_link, publish_time, video_link, view_num_text, publisher_link, publisher_name)
        html_codes = html_codes + t_html

    return html_codes


def get_rec_videos(video_id=None):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "client_secret_774470031846-9nraa6iu1jlkt7bsest4kbr0onud83e5.apps.googleusercontent.com.json"

    credentials = None
    if os.path.exists('oauth_key.pickle'):
        with open('oauth_key.pickle', 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes)
            credentials = flow.run_console()
        # Save the credentials for the next run
        with open('oauth_key.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

    request = youtube.search().list(
        part="snippet",
        maxResults=5,
        relatedToVideoId="jIdq3QMbHCI",
        type="video"
    )
    response = request.execute()
    print(response)
    html_file = process_video_info(snippet=response)

    return html_file

if __name__ == "__main__":
    # main()
    get_rec_videos('IIOH2sCW13U')