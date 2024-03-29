# Sequence dataset, training parameters.

params_ae = {
    'ae_class': 'Autoencoder',
    'train_class': 'train_ae',
    'dataset_folder_name': 'SequenceMemory',
    'log_name': 'minigrid_sequence_memory_ae',
    'latent_dims': 32,
    'input_dims_h': 48,
    'input_dims_w': 48*3,
    'hidden_size': 32,
    'batch_size': 1024*5,
    'epochs': 250,
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

params_vae = {
    'ae_class': 'VariationalAutoencoder',
    'train_class': 'train_vae',
    'dataset_folder_name': 'SequenceMemory',
    'log_name': 'minigrid_sequence_memory_vae',
    'latent_dims': 32,
    'input_dims_h': 48,
    'input_dims_w': 48*3,
    'hidden_size': 32,
    'batch_size': 1024*5,
    'epochs': 250,
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

params_list = [
    params_ae, params_vae]
