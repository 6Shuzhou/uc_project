import random
import shutil 

with open(r"multi-temporal-crop-classification\training_data.txt", "r") as file:
    training_content = file.read()

with open(r"multi-temporal-crop-classification\validation_data.txt", "r") as file:
    validation_content = file.read()

all_training_chips = training_content.split('\n')[:-1]
all_validation_chips = validation_content.split('\n')[:-1]

subset_training_chips = random.sample(all_training_chips, int(len(all_training_chips) * 0.1))
subset_validation_chips = random.sample(all_validation_chips, int(len(all_validation_chips) * 0.1))

source_path_training = "multi-temporal-crop-classification\\training_chips\\"
destination_path_training = "multi-temporal-crop-classification\\training_chips_shrunken\\"
source_path_validation = "multi-temporal-crop-classification\\validation_chips\\"
destination_path_validation = "multi-temporal-crop-classification\\validation_chips_shrunken\\"

for chip in subset_training_chips:
    print(f"Training Chip: {chip}")
    shutil.copyfile(source_path_training + chip + "_merged.tif", destination_path_training + chip + "_merged.tif")
    shutil.copyfile(source_path_training + chip + ".mask.tif", destination_path_training + chip + ".mask.tif")

print('\n')

for chip in subset_validation_chips:
    print(f"Validation Chip: {chip}")
    shutil.copyfile(source_path_validation + chip + "_merged.tif", destination_path_validation + chip + "_merged.tif")
    shutil.copyfile(source_path_validation + chip + ".mask.tif", destination_path_validation + chip + ".mask.tif")
