# Remote job submit: All nodes needs the same conda environments installed.
# In the main server, to create head node, run the following command:
# ray start --head --address "192.168.0.2:6379"
# In the other devices, to create worker nodes, run the following:
# ray start --address "192.168.0.2:6379"
# To start a job, run the following
# ray job submit --address "192.168.0.2:6379" --runtime-env-json {"working-dir": ".", "conda": "pomdp"} -- python visual_pomdp_smm/training/start/start_uniform_memory_training_ray.py

if __name__ == "__main__":
    from visual_pomdp_smm.training.train_utils import start_ray_training

    params_ae = {
        'ae_class': 'Autoencoder',
        'train_class': 'train_ae',
        'dataset_folder_name': 'UniformMemory',
        'log_name': 'minigrid_uniform_memory_ae',
        'latent_dims': 256,
        'input_dims_h': 40,
        'input_dims_w': 40,
        'hidden_size': 128,
        'batch_size': 1024,
        'epochs': 5000,
        'train_set_ratio': 0.8,
        'in_channels': 3,
        'learning_rate': 1e-6,
        'maximum_gradient': 1e7,
        'kernel_size': 7,
        'padding': 0,
        'dilation': 1,
        'conv_hidden_size': 32,
        'conv1_stride': 1,
        'maxpool_stride': 1,
        'lambda': 0.1,
        'gpu_id': 0,
    }
    start_ray_training(params_ae)
