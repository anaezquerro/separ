from argparse import ArgumentParser
import os, shutil, torch 
import torch.distributed as dist

from separ import Parser, DEP_PARSERS, CON_PARSERS, SDP_PARSERS, TAGGERS, DistributedParser
from separ.utils import Config, set_seed, filename

PARSER = {_parser.NAME: _parser for _parser in DEP_PARSERS + SDP_PARSERS + CON_PARSERS + TAGGERS}

def setup():
    dist.init_process_group("nccl")
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    return device

def cleanup():
    dist.destroy_process_group()
    
    
def build_parser(args, dist_mode: bool) -> Parser:
    if dist_mode:
        change_bases(PARSER[args.parser])
    builder = PARSER[args.parser] 
    if args.load:
        try:
            parser = builder.load(args.load, args.device)
            return parser, None 
        except:
            pass 
    assert os.path.exists(args.conf), f'Configuration file does not exist: {args.conf}'
    shutil.copyfile(args.conf, f'{args.output_folder}/{filename(args.conf)}')
    os.makedirs(args.output_folder, exist_ok=True)
    conf = Config.from_ini(args.conf)
    conf.update(vars(args))
    parser = builder.build(**conf, data=args.train)
    return parser, conf
    
    
def run(args, dist_mode: bool = False):
    parser, conf = build_parser(args, dist_mode)
    
    if args.mode == 'train':
        conf['train_conf'].update(vars(args))
        parser.train(**conf['train_conf'])
    elif args.mode == 'eval':
        parser.evaluate(args.input, **vars(args))
    elif args.mode == 'predict':
        parser.predict(args.input, **vars(args))
    
def dist_run(args):
    args.device = setup() 
    run(args, dist_mode=True)
    cleanup()
    
def change_bases(base: type):
    for b in base.__bases__:
        if b != Parser:
            b.__bases__ = change_bases(b)
        elif b == Parser and base != DistributedParser:
            base.__bases__ = (DistributedParser,)
            break 
    return base.__bases__
        
if __name__ == '__main__':
    argparser = ArgumentParser(description='Syntactic Parser')
    subparsers = argparser.add_subparsers(title='Select parser', dest='parser')
    
    for name, parser in PARSER.items():
        subparser = subparsers.add_parser(name)
        subparser = parser.add_arguments(subparser)
        
        subsubparsers = subparser.add_subparsers(title='Run mode', dest='mode')
        train = subsubparsers.add_parser('train')
        evaluate = subsubparsers.add_parser('eval')
        predict = subsubparsers.add_parser('predict')
        
        # train parser 
        train.add_argument('--train', type=str, help='Path to train set')
        train.add_argument('--dev', type=str, help='Path to dev set')
        train.add_argument('--test', type=str, nargs='*', help='Path to test set')
        train.add_argument('-o', '--output-folder', type=str, help='Folder to store training results')
        train.add_argument('--run-name', type=str, default=None, help='Running name for wandb')

        
        # eval parser
        evaluate.add_argument('input', type=str, help='Evaluation dataset')
        evaluate.add_argument('--output', type=str, help='Output folder to store metric')
        evaluate.add_argument('--batch-size', type=int, default=500, help='Inference batch size')

        
        # predict parser 
        predict.add_argument('input', type=str, help='Evaluation dataset')
        predict.add_argument('output', type=str, help='Output dataset')
        predict.add_argument('--batch-size', type=int, default=500, help='Inference batch size')
        
    args = argparser.parse_args()
    set_seed(args.seed)

    if args.device == -1:
        dist_run(args)
    else:
        run(args)
    
