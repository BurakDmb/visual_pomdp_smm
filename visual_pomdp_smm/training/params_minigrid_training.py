# Legacy dataset trainings. Recent is uniform training.

params_minigrid_ae_256 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridDataset',
    'log_name': 'minigrid_AE',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 128,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params_minigrid_ae_64 = params_minigrid_ae_256.copy()
params_minigrid_ae_64['latent_dims'] = 64

params_minigrid_ae_16 = params_minigrid_ae_256.copy()
params_minigrid_ae_16['latent_dims'] = 16

params_minigrid_vae_256 = {
    'ae_class': 'VariationalAutoencoder',
    'train_class': 'train_vae',
    'dataset_class': 'MinigridDataset',
    'log_name': 'minigrid_VAE',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 128,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params_minigrid_vae_64 = params_minigrid_vae_256.copy()
params_minigrid_vae_64['latent_dims'] = 64

params_minigrid_vae_16 = params_minigrid_vae_256.copy()
params_minigrid_vae_16['latent_dims'] = 16


params_memory_binary_ae_256 = {
    'ae_class': 'ConvBinaryAutoencoder',
    'train_class': 'train_ae_binary',
    'dataset_class': 'MinigridMemoryKeySplittedDataset',
    'log_name': 'minigrid_memory_binary_AE',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 128,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params_memory_binary_ae_64 = params_memory_binary_ae_256.copy()
params_memory_binary_ae_64['latent_dims'] = 64

params_memory_binary_ae_16 = params_memory_binary_ae_256.copy()
params_memory_binary_ae_16['latent_dims'] = 16


params_memory_ae_256 = {
    'ae_class': 'ConvAutoencoder',
    'train_class': 'train_ae',
    'dataset_class': 'MinigridMemoryKeySplittedDataset',
    'log_name': 'minigrid_memory_AE',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 128,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-5,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params_memory_ae_64 = params_memory_ae_256.copy()
params_memory_ae_64['latent_dims'] = 64

params_memory_ae_16 = params_memory_ae_256.copy()
params_memory_ae_16['latent_dims'] = 16

params_list = [
    params_minigrid_ae_256,
    params_minigrid_vae_256,
    params_memory_ae_256,
    params_memory_binary_ae_256]

params_list_minigrid_ae_compare_latent = [
    params_minigrid_ae_256,
    params_minigrid_ae_64,
    params_minigrid_ae_16]

params_list_minigrid_vae_compare_latent = [
    params_minigrid_vae_256,
    params_minigrid_vae_64,
    params_minigrid_vae_16]

params_list_binary_ae_compare_latent = [
    params_memory_binary_ae_256,
    params_memory_binary_ae_64,
    params_memory_binary_ae_16]

params_list_memory_ae_compare_latent = [
    params_memory_ae_256,
    params_memory_ae_64,
    params_memory_ae_16]
