from visual_pomdp_smm.testing.minigrid_test import test_function

test_function(
    test_dataset_class_str="MinigridDynamicObsUniformDatasetNoteval",
    eval_dataset_class_str="MinigridDynamicObsUniformDatasetEval",
    prefix_name_inputs="minigrid_uniform_dynamicobs_conv_binary_ae"
)
