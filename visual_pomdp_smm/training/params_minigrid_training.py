# Legacy dataset trainings. Recent is uniform training.

params_minigrid_ae_256 = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_folder_name': 'MinigridKey',
    'log_name': 'minigrid_AE_256',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 1024,
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

params_minigrid_ae_64 = params_minigrid_ae_256.copy()
params_minigrid_ae_64['latent_dims'] = 64
params_minigrid_ae_64['log_name'] = 'minigrid_AE_64'

params_minigrid_ae_16 = params_minigrid_ae_256.copy()
params_minigrid_ae_16['latent_dims'] = 16
params_minigrid_ae_16['log_name'] = 'minigrid_AE_16'

params_minigrid_vae_256 = {
    'ae_class': 'VariationalAutoencoder',
    'train_class': 'train_vae',
    'dataset_folder_name': 'MinigridKey',
    'log_name': 'minigrid_VAE',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 1024,
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

params_minigrid_vae_64 = params_minigrid_vae_256.copy()
params_minigrid_vae_64['latent_dims'] = 64
params_minigrid_vae_64['log_name'] = 'minigrid_VAE_64'

params_minigrid_vae_16 = params_minigrid_vae_256.copy()
params_minigrid_vae_16['latent_dims'] = 16
params_minigrid_vae_16['log_name'] = 'minigrid_VAE_16'


params_memory_conv_binary_ae_256 = {
    'ae_class': 'ConvBinaryAutoencoder',
    'train_class': 'train_ae_binary',
    'dataset_folder_name': 'MinigridKey',
    'log_name': 'minigrid_memory_conv_binary_AE_256',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 1024,
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

params_memory_conv_binary_ae_128 = params_memory_conv_binary_ae_256.copy()
params_memory_conv_binary_ae_128[
    'latent_dims'] = 128
params_memory_conv_binary_ae_128[
    'log_name'] = 'minigrid_memory_conv_binary_AE_128'

params_memory_conv_binary_ae_64 = params_memory_conv_binary_ae_256.copy()
params_memory_conv_binary_ae_64[
    'latent_dims'] = 64
params_memory_conv_binary_ae_64[
    'log_name'] = 'minigrid_memory_conv_binary_AE_64'

params_memory_conv_binary_ae_32 = params_memory_conv_binary_ae_256.copy()
params_memory_conv_binary_ae_32[
    'latent_dims'] = 32
params_memory_conv_binary_ae_32[
    'log_name'] = 'minigrid_memory_conv_binary_AE_32'

params_memory_conv_binary_ae_16 = params_memory_conv_binary_ae_256.copy()
params_memory_conv_binary_ae_16[
    'latent_dims'] = 16
params_memory_conv_binary_ae_16[
    'log_name'] = 'minigrid_memory_conv_binary_AE_16'


params_memory_conv_ae_256 = {
    'ae_class': 'ConvAutoencoder',
    'train_class': 'train_ae',
    'dataset_folder_name': 'MinigridKey',
    'log_name': 'minigrid_memory_conv_AE_256',
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 16,
    'batch_size': 1024,
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

params_memory_conv_ae_128 = params_memory_conv_ae_256.copy()
params_memory_conv_ae_128[
    'latent_dims'] = 128
params_memory_conv_ae_128[
    'log_name'] = 'minigrid_memory_conv_AE_128'

params_memory_conv_ae_64 = params_memory_conv_ae_256.copy()
params_memory_conv_ae_64[
    'latent_dims'] = 64
params_memory_conv_ae_64[
    'log_name'] = 'minigrid_memory_conv_AE_64'

params_memory_conv_ae_32 = params_memory_conv_ae_256.copy()
params_memory_conv_ae_32[
    'latent_dims'] = 32
params_memory_conv_ae_32[
    'log_name'] = 'minigrid_memory_conv_AE_32'

params_memory_conv_ae_16 = params_memory_conv_ae_256.copy()
params_memory_conv_ae_16[
    'latent_dims'] = 16
params_memory_conv_ae_16[
    'log_name'] = 'minigrid_memory_conv_AE_16'

params_list = [
    params_memory_conv_ae_256,
    params_memory_conv_binary_ae_256]

params_list_minigrid_ae_compare_latent = [
    params_minigrid_ae_256,
    params_minigrid_ae_64,
    params_minigrid_ae_16]

params_list_minigrid_vae_compare_latent = [
    params_minigrid_vae_256,
    params_minigrid_vae_64,
    params_minigrid_vae_16]

params_list_binary_ae_compare_latent = [
    params_memory_conv_binary_ae_256,
    params_memory_conv_binary_ae_128,
    params_memory_conv_binary_ae_64,
    params_memory_conv_binary_ae_32,
    params_memory_conv_binary_ae_16]

params_list_memory_ae_compare_latent = [
    params_memory_conv_ae_256,
    params_memory_conv_ae_128,
    params_memory_conv_ae_64,
    params_memory_conv_ae_32,
    params_memory_conv_ae_16]
