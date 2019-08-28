import xml.etree.ElementTree as ET
from os import getcwd
import os

save_folder = 'model_data\\'
dataset_train = 'CSGO_images\\'
dataset_file = save_folder+'NO_CLASSES.txt'
classes_file = dataset_file[:-4]+'_classes.txt'


CLS = os.listdir(dataset_train)
classes =[dataset_train+CLASS for CLASS in CLS]
wd = getcwd()

CLASSES = []
def GetClassesNames(fullname):
    global CLASSES
    in_file = open(fullname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    for i, obj in enumerate(root.iter('object')):
        name = obj.find('name').text
        if name not in CLASSES:
            CLASSES.append(name)

def test(fullname):
    bb = ""
    in_file = open(fullname)
    tree=ET.parse(in_file)
    root = tree.getroot()
    filename = root.find('filename').text
    for i, obj in enumerate(root.iter('object')):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLS or int(difficult)==1:
            continue
        cls_id = CLS.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb += (" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    if bb != "":
        list_file = open(dataset_file, 'a')
        file_string = str(fullname)[:-4]+filename[-4:]+bb+'\n'
        list_file.write(file_string)
        list_file.close()



for CLASS in classes:
    if not CLASS.endswith('.xml'):
        continue
    fullname = os.getcwd()+'\\'+CLASS#+'\\'+filename
    print(fullname)
    GetClassesNames(fullname)

CLASSES.sort()
CLS = CLASSES.copy()
print(CLS)

for CLASS in classes:
    if not CLASS.endswith('.xml'):
        continue
    fullname = os.getcwd()+'\\'+CLASS#+'\\'+filename
    test(fullname)

for CLASS in CLS:
    list_file = open(classes_file, 'a')
    file_string = str(CLASS)+"\n"
    list_file.write(file_string)
    list_file.close()
