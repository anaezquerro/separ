import os, sys, re
from typing import List 
sys.path.append('../')
from separ.utils import listdir, remove

ZIP_PATH = 'penn_treebank.zip'
PARSED_PATH = 'penn_treebank/parsed/mrg/wsj/'
TRAIN_SECTIONS = range(2, 22)
DEV_SECTIONS = [22]
TEST_SECTIONS = [23]
PATH = 'ptb'


def parse_mrg(path: str) -> List[str]:
    lines = open(path, 'r').read().split('\n')
    trees = ['']
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue 
        if re.sub(r'\s+', '', line).startswith('(('):
            trees.append(line)
        else:
            trees[-1] += ' ' + line
    return [re.sub(r' {2,n}', ' ', tree.strip())[1:-1].strip() for tree in trees if len(tree.strip()) > 0]
                           

if __name__ == '__main__':
    if os.path.exists('penn_treebank'):
        os.system('sudo rm -rf penn_treebank')
    remove(PATH)
    os.makedirs(PATH)
    os.system(f'unzip {ZIP_PATH}')
    train, dev, test = [], [], []
    
    for section in sorted(listdir(PARSED_PATH)):
        if not os.path.isdir(f'{PARSED_PATH}/{section}') or not section.isdigit():
            continue
        # select split 
        if int(section) in TRAIN_SECTIONS:
            active = train 
        elif int(section) in DEV_SECTIONS:
            active = dev 
        elif int(section) in TEST_SECTIONS:
            active = test 
        else:
            continue 
        print(f'Parsing section {section}')
            
        # parse all files 
        for file in sorted(listdir(f'{PARSED_PATH}/{section}', absolute=True)):
            active += parse_mrg(file)
    with open(f'{PATH}/train.ptb', 'w') as writer:
        writer.write('\n'.join(train).replace(') )', '))').replace('( (', '(('))
    with open(f'{PATH}/dev.ptb', 'w') as writer:
        writer.write('\n'.join(dev).replace(') )', '))').replace('( (', '(('))
    with open(f'{PATH}/test.ptb', 'w') as writer:
        writer.write('\n'.join(test).replace(') )', '))').replace('( (', '(('))
    os.system('sudo rm -rf penn_treebank')