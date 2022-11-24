
from minigrid.envs import DynamicObstaclesEnv, MemoryEnv
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper
from PIL import Image

from visual_pomdp_smm.envs.minigrid.minigrid_utils import (
    MinigridGenericDatasetEval, MinigridGenericDatasetNoteval)


def main():
    env = MemoryEnv(size=21, agent_view_size=5)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    obs, info = env.reset()
    memory_env_fully_obs_example = Image.fromarray(obs)

    env = DynamicObstaclesEnv(
        size=9, n_obstacles=9*2,
        agent_view_size=5)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    obs, info = env.reset()
    dynamicobs_env_fully_obs_example = Image.fromarray(obs)

    memory_eval_class_data = MinigridGenericDatasetEval(
        "data/", "",
        image_size_h=48,
        image_size_w=48,
        train_set_ratio=0.8,
        dataset_folder_name='UniformMemory',
        use_cache=False)

    memory_test_data = MinigridGenericDatasetNoteval(
        "data/", "",
        image_size_h=48,
        image_size_w=48,
        train_set_ratio=0.8,
        dataset_folder_name='UniformMemory',
        use_cache=False)

    dynamicobs_eval_class_data = MinigridGenericDatasetEval(
        "data/", "",
        image_size_h=48,
        image_size_w=48,
        train_set_ratio=0.8,
        dataset_folder_name='UniformDynamicObs',
        use_cache=False)

    dynamicobs_test_data = MinigridGenericDatasetNoteval(
        "data/", "",
        image_size_h=48,
        image_size_w=48,
        train_set_ratio=0.8,
        dataset_folder_name='UniformDynamicObs',
        use_cache=False)

    img_1_1 = Image.fromarray(memory_eval_class_data.imgs[0])
    img_1_2 = Image.fromarray(memory_test_data.imgs[10])
    img_2_1 = Image.fromarray(dynamicobs_eval_class_data.imgs[0])
    img_2_2 = Image.fromarray(dynamicobs_test_data.imgs[-1])

    img_3_1 = Image.fromarray(memory_eval_class_data.imgs[1])
    img_3_2 = Image.fromarray(memory_test_data.imgs[11])
    img_4_1 = Image.fromarray(dynamicobs_eval_class_data.imgs[1])
    img_4_2 = Image.fromarray(dynamicobs_test_data.imgs[-2])

    img_5_1 = Image.fromarray(memory_eval_class_data.imgs[2])
    img_5_2 = Image.fromarray(memory_test_data.imgs[13])
    img_6_1 = Image.fromarray(dynamicobs_eval_class_data.imgs[2])
    img_6_2 = Image.fromarray(dynamicobs_test_data.imgs[-3])

    # Save images
    memory_env_fully_obs_example.save(
        'results/Example_Fully_Obs_Memory_Env.png')
    dynamicobs_env_fully_obs_example.save(
        'results/Example_Fully_Obs_DynamicObs_Env.png')
    img_1_1.save("results/Example_Memory_Samples_1_1.png")
    img_1_2.save("results/Example_Memory_Samples_1_2.png")
    img_2_1.save("results/Example_Memory_Samples_2_1.png")
    img_2_2.save("results/Example_Memory_Samples_2_2.png")
    img_3_1.save("results/Example_Memory_Samples_3_1.png")
    img_3_2.save("results/Example_Memory_Samples_3_2.png")
    img_4_1.save("results/Example_Memory_Samples_4_1.png")
    img_4_2.save("results/Example_Memory_Samples_4_2.png")
    img_5_1.save("results/Example_Memory_Samples_5_1.png")
    img_5_2.save("results/Example_Memory_Samples_5_2.png")
    img_6_1.save("results/Example_Memory_Samples_6_1.png")
    img_6_2.save("results/Example_Memory_Samples_6_2.png")


if __name__ == "__main__":
    main()
