import os, shutil, sys 
sys.path.append('../')
from separ.data import EnhancedCoNLL
from separ.utils import listdir

TREEBANKS = {
    'arabic': 'ar',
    'bulgarian': 'bg',
    'czech': 'cs',
    'english': 'en',
    'estonian': 'et',
    'finnish': 'fi',
    'french': 'fr',
    'italian': 'it',
    'latvian': 'lv',
    'lithuanian': 'lt',
    'dutch': 'nl',
    'polish': 'pl',
    'russian': 'ru',
    'slovak': 'sk',
    'swedish': 'sv',
    'tamil': 'ta',
    'ukrainian': 'uk',
}
    
SETS = ['train', 'dev', 'test']
FOLDER = 'iwpt-2021'

def remove(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

if __name__ == '__main__':
    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.makedirs(FOLDER, exist_ok=True)
    os.system(f'tar -xvzf iwpt2021stdata.tgz -C {FOLDER}/') # decompress official dataset
    
    for folder in listdir(FOLDER, absolute=True):
        if 'UD_' not in folder:
            remove(folder)
            continue 
        files = list(filter(lambda file: file.endswith('.conllu'), listdir(folder, absolute=True)))
        if len(files) < 3:
            remove(folder)
            continue 
        lang, annt = folder.split('/')[-1].removeprefix('UD_').lower().split('-')
        os.makedirs(f'{FOLDER}/{TREEBANKS[lang]}-{annt}')
        for file in files:
            os.rename(file, f'{FOLDER}/{TREEBANKS[lang]}-{annt}/' + file.split('-')[-1])
        remove(folder)
    
    # os.system(f'tar -czvf iwpt-2021.tar.gz iwpt-2021')
            