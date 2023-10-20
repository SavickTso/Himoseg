import numpy as np
from sklearn.preprocessing import LabelEncoder


label_arr = np.load("movilabels_original.npy")
label_encoder = LabelEncoder()
label_flat = label_arr.flatten()
encoded_label = label_encoder.fit_transform(label_flat)
label_list = list(encoded_label)

ctr_list = np.zeros(max(label_list) + 1)
for i in encoded_label:
    ctr_list[i] += 1


print(ctr_list)

label_to_class = {i: label for i, label in enumerate(label_encoder.classes_)}

print(label_to_class)

print("start compute the classes that not valid for training")
# for idx, ctr in enumerate(ctr_list):
#     if ctr < 9 or ctr > 86:
#         print(label_to_class[idx])

for idx, ctr in enumerate(ctr_list):
    if ctr < 9:
        print(label_to_class[idx], ctr)

exclude_class_list = [
    "bicep_curls_rm",
    "coughing_rm",
    "dj-ing_rm",
    "dribbling_rm",
    "fencing_rm",
    "front_swimming_rm",
    "hopping_rm",
    "juggling_rm",
    "lunges_rm",
    "punching_rm",
    "pushups_rm",
    "rowing_rm",
    "serving_rm",
    "stretching_rm",
    "wearing_belt_rm",
    "yoga_rm",
    "free_throw_rm",
    "throwing_frisbee_rm",
    "swinging_arms_rm",
    "swinging_racket_rm",
]

print(len(exclude_class_list))
