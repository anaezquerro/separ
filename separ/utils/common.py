import torch, os 

try:
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
except:
    WORLD_SIZE = torch.cuda.device_count()
NUM_WORKERS = 1