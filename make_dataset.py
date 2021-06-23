"""
module for making preprocessed labels
and image matrices of NWPU-Crowd Dataset
"""

from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import requests
import pickle
import os
import numpy as np
import tensorflow as tf

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def make_xy():
    """
    function for fetching json files of 3,609 images
    and importing their matrix and labels
    """

    # connecting google drive
    print("connecting to google drive... \n")
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secrets.json", SCOPES
            )
            creds = flow.run_local_server(port=8080)
        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    # importing (id, label) and (id, image) data by looping over
    access_token = creds.token
    page_token_json = None
    page_token_img = None
    json_ids = []
    img_ids = []

    # getting google drive ids of json files
    while True:
        url = "https://www.googleapis.com/drive/v3/files/"
        resp_json_id = requests.get(
            url,
            headers={"Authorization": "Bearer " + access_token},
            params={
                "q": "mimeType='application/json' and '1qfpt_6CVJye5ca1rI2ZY0KWwsX3-s3z8' in parents",
                "pageToken": page_token_json,
                "pageSize": "1000",
            },
        )
        json_id = resp_json_id.json()
        page_token_json = json_id.get("nextPageToken")
        for file in json_id.get("files"):
            json_ids.append(file.get("id"))

        if not page_token_json:
            break

    # getting google drive ids for image files
    while True:
        url = "https://www.googleapis.com/drive/v3/files/"
        resp_img_id = requests.get(
            url,
            headers={"Authorization": "Bearer " + access_token},
            params={
                "q": "('1c0x3fhYJDytUO-73FN70Pe-wFKUidNzo' in parents or '1ZrbPLeNJc1qnVhnROcpW6UIiimeLmKu5' in parents or '12R6kvBkbw0k__eWfHVZLDKlNPrnSjoIK' in parents or '1OW0BDFc4zqUvgSL35vmJ8Zqj9WGJnota' in parents)",
                "pageToken": page_token_img,
                "pageSize": "1000",
            },
        )
        img_id = resp_img_id.json()
        page_token_img = img_id.get("nextPageToken")
        for file in img_id.get("files"):
            img_ids.append(file.get("id"))

        if not page_token_img:
            break

    def read_label(id):
        """
        inner function for multiprocessing ; getting image ids and labels
        """
        url = "https://www.googleapis.com/drive/v3/files/"
        resp_label = requests.get(
            url + id + "?alt=media",
            headers={"Authorization": "Bearer " + access_token},
        )
        json_label = resp_label.json()
        return (json_label.get("img_id"), json_label.get("human_num"))

    def read_img(id):
        """
        inner function for multiprocessing ; getting image ids and image matrices
        """
        url = "https://www.googleapis.com/drive/v3/files/"
        resp_img_id = requests.get(
            url + id,
            headers={"Authorization": "Bearer " + access_token},
        )
        resp_img = requests.get(
            url + id + "?alt=media",
            headers={"Authorization": "Bearer " + access_token},
        )
        img_id = resp_img_id.json().get("name").split(".")[0]
        img = tf.image.decode_jpeg(resp_img.content, channels=3)
        img = tf.image.resize(img, [224, 224]) / 255

        return (img_id, img[None, :, :, :])

    print("preprocessing labels... \n")
    pool = ThreadPool(20)
    label_list = [
        _ for _ in tqdm(pool.imap_unordered(read_label, json_ids), total=len(json_ids))
    ]
    pool.close()
    pool.join()

    print("preprocessing images... \n")
    pool = ThreadPool(20)
    img_list = [
        _ for _ in tqdm(pool.imap_unordered(read_img, img_ids), total=len(img_ids))
    ]
    pool.close()
    pool.join()

    # sorting
    label_list.sort(key=lambda data: data[0])
    img_list.sort(key=lambda data: data[0])
    id, y = zip(*label_list)
    id, X = zip(*img_list)
    y = np.array(y).astype(np.float32)
    X = np.concatenate(X, axis=0).astype(np.float32)
    print("\n dataset is loaded")

    return (y, X)
