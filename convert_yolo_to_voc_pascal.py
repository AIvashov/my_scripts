# Script to convert yolo annotations to voc format
# 
# TODO:\\ Description
# 
# Sample format
# <annotation>
#     <folder>example</folder>
#     <filename>example.jpg</filename>
#     <path>C:\User\example\example.jpg</path>
#     <source>
#         <database>Unknown</database>
#     </source>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>Car</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>

import argparse
import os
import xml.etree.cElementTree as ET
from PIL import Image

CLASS_MAPPING = {
    '0' : 'person',
    '1' : 'bicycle',
    '2' : 'car',
    '3' : 'motorcycle',
    '5' : 'bus',
    '7' : 'truck',    
}

def get_image_size(image_path)->str:
    im = Image.open(image_path)
    (width, height) = im.size
    return width, height

def get_image_path_prop(image_path)->str:
    filename = image_path.split('\\')[-1]
    folder = image_path.split('\\')[-2]   
    return filename, folder

def create_annotation_head(image_path) -> str:
    width, height = get_image_size(image_path)
    filename, folder = get_image_path_prop(image_path)
    annotation_head = ET.Element("annotations")
    ET.SubElement(annotation_head, "folder").text = folder
    ET.SubElement(annotation_head, "filename").text = filename
    ET.SubElement(annotation_head, "path").text = image_path
    source = ET.SubElement(annotation_head, "source")
    ET.SubElement(source, "database").text = 'Unknown'
    size = ET.SubElement(annotation_head, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation_head, "segmented").text = '0'
    return annotation_head

def read_txt_file(file_txt_path) -> str:
    with open(file_txt_path) as file:
        lines = file.readlines()
        return lines
    
def convert_coordinates(image_path, lines):
    width, height = get_image_size(image_path)
    voc_labels = []
    for line in lines:
        voc = []
        line = line.strip()
        data = line.split()
        voc.append(CLASS_MAPPING.get(data[0]))
        bbox_width = float(data[3]) * width
        bbox_height = float(data[4]) * height
        center_x = float(data[1]) * width
        center_y = float(data[2]) * height
        voc.append(center_x - (bbox_width / 2))
        voc.append(center_y - (bbox_height / 2))
        voc.append(center_x + (bbox_width / 2))
        voc.append(center_y + (bbox_height / 2))
        voc_labels.append(voc)
    return voc_labels

def create_objects_annotation(annotation_head, voc_labels):
    for voc_label in voc_labels:
        object_annotation = ET.SubElement(annotation_head, "object")
        ET.SubElement(object_annotation, "name").text = voc_label[0]
        ET.SubElement(object_annotation, "pose").text = "Unspecified"
        ET.SubElement(object_annotation, "truncated").text = str(0)
        ET.SubElement(object_annotation, "difficult").text = str(0)
        bbox = ET.SubElement(object_annotation, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(round(voc_label[1]))
        ET.SubElement(bbox, "ymin").text = str(round(voc_label[2]))
        ET.SubElement(bbox, "xmax").text = str(round(voc_label[3]))
        ET.SubElement(bbox, "ymax").text = str(round(voc_label[4]))
    return annotation_head   
    
def create_annotation(image_path, file_txt_path, save_dir):
    annotation_head = create_annotation_head(image_path)
    lines = read_txt_file(file_txt_path)
    voc_labels = convert_coordinates(image_path, lines)
    objects_annotation = create_objects_annotation(annotation_head, voc_labels)
    annotation = ET.ElementTree(objects_annotation)
    filename = file_txt_path.split('\\')[-1][:-4] + '.xml'
    file_path =  os.path.join(save_dir, filename)
    annotation.write(file_path)
    print('Save to ...' + file_path)

def convert_yolo_to_voc(indir_annotation,indir_image,out_dir):
    for file_img in os.listdir(indir_image):
        file_txt = file_img[:-4] + '.txt'
        image_path = os.path.join(indir_image, file_img)
        file_txt_path = os.path.join(indir_annotation, file_txt)
        create_annotation(image_path, file_txt_path, out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO annotation to VOC Pascal')
    parser.add_argument('indir_annotation', type=str, help='Input dir for yolo annotation')
    parser.add_argument('indir_image', type=str, help='Input dir for image')
    parser.add_argument('out_dir', type=str, help='Output dir for save voc pascal annotation')
    args = parser.parse_args()
    print(args)


    INDIR_ANNOTATION =  args.indir_annotation
    INDIR_IMAGE = args.indir_image
    OUT_DIR = args.out_dir
    
    convert_yolo_to_voc(INDIR_ANNOTATION, INDIR_IMAGE, OUT_DIR)