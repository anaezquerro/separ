from configparser import ConfigParser
from argparse import ArgumentParser
import torch, os
import torch.multiprocessing as mp

from trasepar import DEP_PARSERS, CON_PARSERS, SDP_PARSERS, TAGGERS
from trasepar.utils import Config, set_seed, init_folder, NUM_WORKERS

PARSER = {_parser.NAME: _parser for _parser in DEP_PARSERS + CON_PARSERS + SDP_PARSERS + TAGGERS}
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
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
        train.add_argument('--num-workers', type=int, default=1, help='Number of processes')

        
        # eval parser
        evaluate.add_argument('input', type=str, help='Evaluation dataset')
        evaluate.add_argument('--output', type=str, help='Output folder to store metric')
        evaluate.add_argument('--batch-size', type=int, default=100, help='Inference batch_size')
        evaluate.add_argument('--num-workers', type=int, default=10, help='Number of processes')

        
        # predict parser 
        predict.add_argument('input', type=str, help='Evaluation dataset')
        predict.add_argument('output', type=str, help='Output dataset')
        predict.add_argument('--batch-size', type=int, default=100, help='Inference batch size')
        predict.add_argument('--num-workers', type=int, default=10, help='Number of processes')
        
    args = argparser.parse_args()
    args.device = f'cuda:{args.device}'
    torch.cuda.set_device(args.device)
    set_seed(args.seed)
    args.num_workers = NUM_WORKERS or args.num_workers # override num-workers

    if args.mode == 'train' and os.path.exists(args.path):
        rem = input(f'[WARNING]: The path {args.path} has some content that might be removed.\n' +
                    'Are you sure you want to continue? (Ctrl+C to cancel, otherwise press enter)')

    # select parser and build
    constructor = PARSER[args.parser]
    if args.load or args.mode != 'train': # load parser 
        parser = constructor.load(args.path, args.device)
    else:
        init_folder(args.path)
        assert os.path.exists(args.conf), f'Configuration file does not exist: {args.conf}'
        build_params = Config.from_ini(args.conf)
        args.data = constructor.load_data(args.train, num_workers=args.num_workers)
        build_params.update(**vars(args))
        parser = constructor.build(**build_params)
        
    # select mode 
    if args.mode == 'train':
        train_params = vars(args)
        if 'train_conf' in build_params.keys():
            train_params.update(**build_params['train_conf']())
        
        parser.train(**train_params)
    elif args.mode == 'eval':
        parser.model.requires_grad_(False)
        parser.evaluate(args.input, batch_size=args.batch_size, num_workers=args.num_workers, path=args.output)
    elif args.mode == 'predict':
        parser.model.requires_grad_(False)
        parser.evaluate(args.input, args.output, batch_size=args.batch_size, num_workers=args.num_workers)