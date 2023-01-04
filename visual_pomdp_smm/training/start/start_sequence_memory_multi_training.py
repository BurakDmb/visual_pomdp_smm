import torch

from visual_pomdp_smm.training.train_utils import start_multi_training

if __name__ == '__main__':
    from visual_pomdp_smm.training.params.params_sequence_memory_training import \
        params_list
    NUM_GPUS = torch.cuda.device_count()
    PROC_PER_GPU = 6
    N = 12
    start_multi_training(params_list, NUM_GPUS, PROC_PER_GPU, N)
