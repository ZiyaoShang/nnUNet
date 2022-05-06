import tensorflow as tf
import utils

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


generate_num = 100 # number of images to generate for each modality
rounds = 5  # 1 round = 100 epochs + one database refresh

# Step0: setup env variables:
# export nnUNet_raw_data_base="/home/ziyaos/nnunet_ssg/nnUNet_raw_data_base" && export nnUNet_preprocessed="/home/ziyaos/nnunet_ssg/nnUNet_preprocessed" && export RESULTS_FOLDER="/home/ziyaos/nnunet_ssg/nnUNet_trained_models"

print("Step1: generating dataset...")
utils.generate_dataset(generate_num,
                 merge_labels=np.array([([0], 0), ([257, 4, 43, 14, 15, 28, 60, 24, 30, 31, 44, 62, 63, 5], 1),
                ([2, 21, 41, 61, 7, 46, 16, 170, 77, 85], 2), ([3, 42, 8, 47, 172, 9, 48, 10, 49,
                11, 50, 12, 51, 17, 53, 18, 54, 13, 52, 26, 58], 3)]))

print("Step2: generating data.json...")
utils.generate_json()

print("Step3: don't forget to generate plan for the first round...")
# nnUNet_plan_and_preprocess.main()
# nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity

print("Step4: start 1st round training...")
utils.train(continue_training=False)

# kill in the middle + continue == works and continues
print('done')


