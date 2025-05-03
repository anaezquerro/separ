import os, shutil, sys, re
sys.path.append('../')
from separ.utils import listdir, remove

FOLDER = 'spmrl-2014'
PATH = 'spmrl-2014.tar.gz'
LANG = {'basque': 'eu', 'french': 'fr', 'german': 'de', 'hebrew': 'he', 'hungarian': 'hu', 'korean': 'ko', 
        'polish': 'pl', 'swedish': 'sv'}
remove(FOLDER)
remove(PATH)
    
def parse_ptb(path: str):
    lines = open(path, 'r').read().strip().split('\n')
    for i, line in enumerate(lines):
        if line.replace(' ', '').startswith('(('):
            lines[i] = line.strip()[1:-1].strip()
    with open(path, 'w') as writer:
        writer.write('\n'.join(lines))
    
def parse_subsets(path: str):
    if not os.path.exists(f'{path}/gold/ptb/train'):
        os.rename(f'{path}/gold/ptb/train5k', f'{path}/gold/ptb/train')
    for subset in ['train', 'dev', 'test']:
        for file in listdir(f'{path}/gold/ptb/{subset}', absolute=True):
            if file.endswith('.ptb'):
                os.rename(file, f'{path}/{subset}.ptb')
                parse_ptb(f'{path}/{subset}.ptb')
    for sub in listdir(path, absolute=True):
        if not sub.endswith('.ptb'):
            remove(sub)
    
            

if __name__ == '__main__':
    os.system(f'unzip SPMRL_SHARED_2014_NO_ARABIC.zip')
    os.system(f'mv SPMRL_SHARED_2014_NO_ARABIC {FOLDER}')
    for file in listdir(FOLDER):
        if file.endswith('.tar.gz'):
            os.system(f'tar -xvf {FOLDER}/{file} -C {FOLDER}/')
            remove(f'{FOLDER}/{file}')
    for file in listdir(FOLDER):
        parse_subsets(f'{FOLDER}/{file}')
        name, *_ = file.split('_')
        os.rename(f'{FOLDER}/{file}', f'{FOLDER}/{LANG[name.lower()]}')
    # os.system(f'tar -czvf {PATH} {FOLDER}')            
            