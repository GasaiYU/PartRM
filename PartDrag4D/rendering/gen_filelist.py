import os
import re

base = '../data/processed_data_partdrag4d'


if __name__ == '__main__':
    classes = ['Dishwasher', 'Laptop', 'Microwave', 'Oven', 
               'Refrigerator', 'StorageFurniture', 'WashingMachine', 'TrashCan']
               
    with open('../filelist/rendering.txt', 'w') as f:
        for class_name in classes:
            path = os.path.join(base, class_name)
            for root, dirs, files in os.walk(path):
                for file in files:
                    if 'motion' in root:
                        item_name = os.path.normpath(root).split(os.sep)[-2]
                        item_idx = re.search(r'\d+', item_name).group() + '_' + file[0]
                        if file.endswith('.obj'):
                            f.write(os.path.join(root, file) + '\n')
    print('done')