from configparser import ConfigParser
from argparse import ArgumentParser
import torch, os, shutil, sys
import torch.multiprocessing as mp

from separ import DEP_PARSERS, CON_PARSERS, SDP_PARSERS
from separ.utils import Config, set_seed, init_folder, filename
from separ.utils.common import NUM_THREADS

PARSER = {_parser.NAME: _parser for _parser in DEP_PARSERS + CON_PARSERS + SDP_PARSERS}
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    argparser = ArgumentParser(description='Initialize Parser')
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
        train.add_argument('--seed', type=int, default=123, help='Training seed')
        train.add_argument('--num-workers', type=int, default=NUM_THREADS, help='Number of processes')

        
        # eval parser
        evaluate.add_argument('input', type=str, help='Evaluation dataset')
        evaluate.add_argument('--batch-size', type=int, default=100, help='Inference batch_size')
        evaluate.add_argument('--num-workers', type=int, default=NUM_THREADS, help='Number of processes')

        
        # predict parser 
        predict.add_argument('input', type=str, help='Evaluation dataset')
        predict.add_argument('output', type=str, help='Output dataset')
        predict.add_argument('--batch-size', type=int, default=100, help='Inference batch size')
        predict.add_argument('--num-workers', type=int, default=NUM_THREADS, help='Number of processes')
        
    args = argparser.parse_args()
    args.device = f'cuda:{args.device}'
    torch.cuda.set_device(args.device)
    set_seed(args.seed)

    # select parser and build
    constructor = PARSER[args.parser]
    if args.load: # load parser 
        parser = constructor.load(args.path, f'cuda:{args.device}')
    else:
        init_folder(args.path)
        config = ConfigParser()
        config.read(args.conf)
        build_params = {conf: Config.from_ini(config[conf]) for conf in config.sections()}
        args.data = constructor.DATASET.from_file(args.train, num_workers=args.num_workers)
        build_params.update(**vars(args))
        parser = constructor.build(**build_params)
    
    # select mode 
    if args.mode == 'train':
        train_params = vars(args)
        if 'train_conf' in build_params.keys():
            train_params.update(**build_params['train_conf']())
        
        parser.train(**train_params)
        shutil.copyfile(args.conf, f'{args.path}/{filename(args.conf)}') # copy the configuration file
    elif args.mode == 'eval':
        parser.model.requires_grad_(False)
        parser.evaluate(args.input, batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.mode == 'predict':
        parser.model.requires_grad_(False)
        parser.evaluate(args.input, args.output, batch_size=args.batch_size, num_workers=args.num_workers)


        
        
    
    
        
        
    
    
