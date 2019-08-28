from textwrap import dedent
from lxml import etree
import glob
import os
import cv2
import time

# input is result of cv.imread(image)
def CreateXMLfile(image, ObjectsList):
    Folder_to_save = "XML_images"
    os.chdir(Folder_to_save)
    image_num = len(glob.glob("*.jpg"))
    print("Number of .jpg files =", image_num)

    img_name = "XML_"+str(image_num+1)+".jpg"
    while os.path.isfile(img_name):
        image_num+=1
        img_name = "XML_"+str(image_num)+".jpg"
    
    cv2.imwrite(img_name,image)

    annotation = etree.Element("annotation")

    folder = etree.Element("folder")
    folder.text = os.path.basename(os.getcwd())
    annotation.append(folder)

    filename_xml = etree.Element("filename")
    filename_str = img_name.split(".")[0]
    filename_xml.text = img_name
    annotation.append(filename_xml)

    path = etree.Element("path")
    path.text = os.path.join(os.getcwd(), filename_str + ".jpg")
    annotation.append(path)

    source = etree.Element("source")
    annotation.append(source)

    database = etree.Element("database")
    database.text = "Unknown"
    source.append(database)

    size = etree.Element("size")
    annotation.append(size)

    width = etree.Element("width")
    height = etree.Element("height")
    depth = etree.Element("depth")

    img = cv2.imread(filename_xml.text)

    width.text = str(img.shape[1])
    height.text = str(img.shape[0])
    depth.text = str(img.shape[2])

    size.append(width)
    size.append(height)
    size.append(depth)

    segmented = etree.Element("segmented")
    segmented.text = "0"
    annotation.append(segmented)

    for Object in ObjectsList:
        class_name = Object[6]
        xmin_l = str(int(float(Object[1])))
        ymin_l = str(int(float(Object[0])))
        xmax_l = str(int(float(Object[3])))
        ymax_l = str(int(float(Object[2])))

        obj = etree.Element("object")
        annotation.append(obj)

        name = etree.Element("name")
        name.text = class_name
        obj.append(name)

        pose = etree.Element("pose")
        pose.text = "Unspecified"
        obj.append(pose)

        truncated = etree.Element("truncated")
        truncated.text = "0"
        obj.append(truncated)

        difficult = etree.Element("difficult")
        difficult.text = "0"
        obj.append(difficult)

        bndbox = etree.Element("bndbox")
        obj.append(bndbox)

        xmin = etree.Element("xmin")
        xmin.text = xmin_l
        bndbox.append(xmin)

        ymin = etree.Element("ymin")
        ymin.text = ymin_l
        bndbox.append(ymin)

        xmax = etree.Element("xmax")
        xmax.text = xmax_l
        bndbox.append(xmax)

        ymax = etree.Element("ymax")
        ymax.text = ymax_l
        bndbox.append(ymax)

    # write xml to file
    s = etree.tostring(annotation, pretty_print=True)
    with open(filename_str + ".xml", 'wb') as f:
        f.write(s)
        f.close()

    os.chdir("..")
    time.sleep(2)
