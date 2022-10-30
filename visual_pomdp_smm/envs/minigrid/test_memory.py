from visual_pomdp_smm.testing.minigrid_test import test_function

test_function(
    test_dataset_class_str="MinigridMemoryUniformDatasetNoteval",
    eval_dataset_class_str="MinigridMemoryUniformDatasetEval",
    prefix_name="minigrid_uniform_memory_ae"
)
