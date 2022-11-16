from visual_pomdp_smm.training.train_utils import start_multi_training
import torch


if __name__ == '__main__':
    from visual_pomdp_smm.training.params_uniform_latent_space_training\
        import params_list
    NUM_GPUS = torch.cuda.device_count()
    PROC_PER_GPU = 8
    N = 12
    start_multi_training(params_list, NUM_GPUS, PROC_PER_GPU, N)
