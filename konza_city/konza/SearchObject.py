# import the necessary packages
import numpy as np
import argparse
import pandas as pd
import cv2
from imutils import paths
from django.shortcuts import render
import csv
from .models import *


def countItem(lst, *argv):
    items = {}
    for arg in argv:
        items.update({arg: lst.count(arg)})
        # print(lst.count(arg))
    return items


def query_to_csv(queryset, filename='items.csv', **override):
    field_names = [field.name for field in queryset.model._meta.fields]

    def field_value(row, field_name):
        if field_name in override.keys():
            return override[field_name]
        else:
            return row[field_name]
    with open(filename, 'w+', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter=',')
        writer.writerow(field_names)
        for row in queryset.values(*field_names):
            writer.writerow([field_value(row, field) for field in field_names])


def read_data(path):
    ted_data = pd.read_csv(path)
    return ted_data


class SearchObject:
    def __init__(self, imagePath):
        self.imagePath = imagePath

    def process(self, model="MobileNetSSD_deploy.caffemodel"):
        data = []

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        prototxt = 'MobileNetSSD_deploy.prototxt.txt'
        # ap = argparse.ArgumentParser()

        # ap.add_argument("-p", "--prototxt", default=prototxt,
        #                 help="path to Caffe 'deploy' prototxt file")
        # ap.add_argument("-m", "--model", default=model,
        #                 help="path to Caffe pre-trained model")
        # args = vars(ap.parse_args())

        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(
            'konza/MobileNetSSD_deploy.prototxt.txt', 'konza/MobileNetSSD_deploy.caffemodel')
        q = Post.objects.all().order_by('-created_on')[:1]
        query_to_csv(q, filename='data.csv', user=1, group=1)
        d = read_data('data.csv')
        key = d['image']
        key = key[0]

        image = cv2.imread('media/' + key)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)

        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.2:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # display the prediction
                label = "{}: {:.2f}%".format(
                    CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                # data.append(label)

                data.append(CLASSES[idx])
                # print(data)

                cv2.rectangle(image, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                cv2.imwrite("media/new.jpg", image)

        final = countItem(data, 'bus', 'car', 'train', 'person')

        print(final)
        return final
        # return data

    # show the output image
    # cv2.waitKey(0)
