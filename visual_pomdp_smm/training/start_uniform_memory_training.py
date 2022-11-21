
if __name__ == "__main__":
    from visual_pomdp_smm.training.params_uniform_memory_training\
        import params_list
    from visual_pomdp_smm.training.train_utils import start_training

    import torch.multiprocessing as mp
    mp.set_start_method('forkserver', force=True)

    processes = []
    for params in params_list:
        p = mp.Process(
            target=start_training,
            args=(params,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
