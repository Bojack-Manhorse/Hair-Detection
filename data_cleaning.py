import json

def clean_annotations_file():

    with open('Datasets/face-h 2.v2i.coco/_annotations.coco.json') as f:
        annotations_dictionary = json.load(f)
    
    annotations_dictionary['images'] = [item for item in annotations_dictionary['images'] if item['height'] == 1024]

    annotations_dictionary['annotations'] = [item for item in annotations_dictionary['annotations'] if item['category_id'] in [1,3,8,9]]

    list_of_image_ids = [item['image_id'] for item in annotations_dictionary['annotations']]

    annotations_dictionary['annotations'] = [item for item in annotations_dictionary['annotations'] if list_of_image_ids.count(item['image_id']) == 1]

    annotations_dictionary['images'] = [item for item in annotations_dictionary['images'] if list_of_image_ids.count(item['id']) == 1]

    return annotations_dictionary

if __name__ == "__main__":

    annotations_dictionary = clean_annotations_file()

    with open('Datasets/face-h 2.v2i.coco/processed_annotations.json', 'w+') as f:
        json.dump(annotations_dictionary, f)