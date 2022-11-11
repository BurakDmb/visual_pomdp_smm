# For The GPU Job Distribution, stackoverflow has been used.
# https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool

from visual_pomdp_smm.training.params_minigrid_training import \
    params_list_memory_ae_compare_latent as params_list
from visual_pomdp_smm.training.train_utils import start_training
import torch.multiprocessing as mp
from itertools import repeat
import torch
# from torch.multiprocessing import Pool, Queue


NUM_GPUS = torch.cuda.device_count()
PROC_PER_GPU = 8
N = 12


def mp_pool_function(param, queue):
    gpu_id = queue.get()
    try:
        param['gpu_id'] = gpu_id
        print(
            "Experiment log_name=" + param['log_name'] +
            " (gpu_id=" + str(gpu_id) + ") started.")
        start_training(param)
        print(
            "Experiment log_name=" + param['log_name'] +
            " (gpu_id=" + str(gpu_id) + ") finished.")
    except Exception as e:
        print(e)
    finally:
        queue.put(gpu_id)


def mp_create_experiment_params():
    # Creating experiment parameter sets,
    # while assining each with an unique id.
    experiment_params = []
    for param in params_list:
        for i in range(N):
            param_ = param.copy()
            param_['log_name'] = param_['log_name'] + '_' + str(i)
            experiment_params.append(param_)
    return experiment_params


def main():
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    queue = manager.Queue()
    experiment_params = mp_create_experiment_params()

    # Initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    pool = mp.Pool(processes=PROC_PER_GPU * NUM_GPUS)

    for _ in pool.starmap(mp_pool_function, zip(
            experiment_params, repeat(queue))):
        pass
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
