import os, shutil, sys 
sys.path.append('../')
from separ.data import EnhancedCoNLL, conll_to_sdp
from separ.utils.common import NUM_THREADS
from separ.utils import listdir

TREEBANKS = dict(
    ar = 'arabic',
    bg = 'bulgarian',
    cs = 'czech', 
    en = 'english',
    et = 'estonian',
    fi = 'finnish',
    fr = 'french',
    it = 'italian', 
    lv = 'latvian', 
    lt = 'lithuanian',
    nl = 'dutch', 
    pl = 'polish',
    ru = 'russian',
    sk = 'slovak',
    sv = 'swedish',
    ta = 'tamil', 
    uk = 'ukrainian'
)
    
SETS = ['train', 'dev', 'test']
FOLDER = 'iwpt-2021'


if __name__ == '__main__':
    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.makedirs(FOLDER, exist_ok=True)
    os.system(f'tar -xvzf iwpt2021stdata.tgz -C {FOLDER}/') # decompress official dataset
    
    for code, lang in TREEBANKS.items():
        os.makedirs(f'{FOLDER}/{code}', exist_ok=True)
        for folder in listdir(FOLDER, absolute=True):
            if lang in folder.lower():
                files = filter(lambda file: file.endswith('.conllu'), listdir(folder, absolute=True))
                for file in files:
                    content = open(file).read()
                    new_file = file.split('-')[-1]
                    with open(f'iwpt-2021/{code}/{new_file}', 'a') as writer:
                        writer.write(content)
                shutil.rmtree(folder)
            if folder.split('/')[-1] not in TREEBANKS.keys() and 'UD_' not in folder:
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
                else:
                    os.remove(folder)

        for sett in SETS:
            print(f'Converting {FOLDER}/{code}/{sett}.conllu')
            data = EnhancedCoNLL.from_file(f'{FOLDER}/{code}/{sett}.conllu', num_workers=NUM_THREADS)
            conll_to_sdp(data).save(f'{FOLDER}/{code}/{sett}.sdp')
            data.save(f'{FOLDER}/{code}/{sett}.conllu')
    os.system(f'tar -czvf iwpt-2021.tar.gz iwpt-2021')
            
            
            
            
        
        
                
                