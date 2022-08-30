params256 = {
    'latent_dims': 256,
    'input_dims': 48,
    'hidden_size': 32,
    'batch_size': 1024,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}


params128 = {
    'latent_dims': 128,
    'input_dims': 48,
    'hidden_size': 32,
    'batch_size': 1024,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}


params64 = {
    'latent_dims': 64,
    'input_dims': 48,
    'hidden_size': 32,
    'batch_size': 1024,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params32 = {
    'latent_dims': 32,
    'input_dims': 48,
    'hidden_size': 32,
    'batch_size': 1024,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}

params16 = {
    'latent_dims': 16,
    'input_dims': 48,
    'hidden_size': 32,
    'batch_size': 1024,
    'epochs': 500,
    'train_set_ratio': 0.8,
    'in_channels': 3,
    'learning_rate': 1e-4,
    'maximum_gradient': 1e7,
    'kernel_size': 4,
    'padding': 3,
    'dilation': 1,
    'conv_hidden_size': 32,
    'conv1_stride': 6,
    'maxpool_stride': 1,
    'lambda': 10,
}
params_list = [params256, params128, params64, params32, params16]
params_list = [params256, params64, params16]