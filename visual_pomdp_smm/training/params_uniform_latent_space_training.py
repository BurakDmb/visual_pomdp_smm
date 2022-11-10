# Uniform dataset, training parameters.

params_memory_ae8 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridMemoryUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_memory_8',
    'latent_dims': 8,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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

params_memory_ae32 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridMemoryUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_memory_32',
    'latent_dims': 32,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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

params_memory_ae256 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridMemoryUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_memory_256',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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
params_dynamicobs_ae8 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridDynamicObsUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_dynamicobs_8',
    'latent_dims': 8,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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

params_dynamicobs_ae32 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridDynamicObsUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_dynamicobs_32',
    'latent_dims': 32,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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

params_dynamicobs_ae256 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridDynamicObsUniformDataset',
    'log_name': 'minigrid_uniform_latentspace_dynamicobs_256',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 2048,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
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


# params_list = [
#     params_memory_ae8, params_memory_ae32, params_memory_ae256]

params_list = [
    params_dynamicobs_ae8, params_dynamicobs_ae32, params_dynamicobs_ae256]
