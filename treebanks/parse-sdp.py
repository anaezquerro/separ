import os, shutil 

FOLDER = 'sdp-2015'
TREEBANKS = ['zh', 'dm', 'pas', 'psd', 'cs']
PATH = f'sdp2014_2015/data/2015'


def split(path: str, pattern: str = '#220'):
    """Reads a train.sdp file and splits 20xxxxx sentences for the validation."""
    lines = open(path, 'r').read().split('\n')
    dev = False
    train_lines, dev_lines = [], ['#SDP 2015']
    for line in lines:
        if line.startswith(pattern):
            dev = True 
        elif line.startswith('#'):
            dev = False 
            
        if dev:
            dev_lines.append(line)
        else:
            train_lines.append(line)
        if line == '\n':
            dev = False 
    with open(path, 'w') as writer:
        writer.write('\n'.join(train_lines))
        writer.write('\n\n')
    with open(path.replace('train.sdp', 'dev.sdp'), 'w') as writer:
        writer.write('\n'.join(dev_lines))
        writer.write('\n\n')


if __name__ == '__main__':
    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.makedirs(FOLDER, exist_ok=True)
    os.system(f'tar -xvzf sdp2014_2015_LDC2016T10.tgz')
    
    for treebank in TREEBANKS:
        if treebank == 'zh':
            lang = 'cz' 
            dataset = 'pas'
        elif treebank == 'cs':
            lang = 'cs'
            dataset = 'psd'
        else:
            lang = 'en'
            dataset = treebank
        os.makedirs(f'{FOLDER}/{treebank}')
        os.rename(f'{PATH}/{lang}.{dataset}.sdp', f'{FOLDER}/{treebank}/train.sdp')
        os.rename(f'{PATH}/test/{lang}.id.{dataset}.sdp', f'{FOLDER}/{treebank}/id.sdp')
        if treebank != 'zh':
            os.rename(f'{PATH}/test/{lang}.ood.{dataset}.sdp', f'{FOLDER}/{treebank}/ood.sdp')
        else:
            shutil.copy(f'{FOLDER}/{treebank}/id.sdp', f'{FOLDER}/{treebank}/ood.sdp')
        split(f'{FOLDER}/{treebank}/train.sdp', pattern=('#220' if treebank != 'zh' else '#202'))
    shutil.rmtree('sdp2014_2015')
    # os.system(f'tar -czvf sdp-2015.tar.gz sdp-2015')