import scipy.io as sp
import numpy as np
import csv
import os
import IPython
import re


# Custom key function to extract and sort by the number in the filename
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0  # Default to 0 if no number is found


def main():
    name_list = []
    ctr = 0
    directory_path = "/home/cao/dataset/movi/F_AMASS/AMASS/"
    # List all files in the directory
    file_list = os.listdir(directory_path)
    for file_name in file_list:
        if file_name.endswith(".mat"):
            # Construct the full file path
            name_list.append(file_name)

            # mat_contents1 = sp.loadmat("F_amass_Subject_15.mat")
            # mat_contents1["Subject_15_F_amass"]["move"][0][0][0][0][0]["description"][0][0]

    sorted_filenames = sorted(name_list, key=extract_number)
    print(sorted_filenames)
    # Extract and print the integers from the sorted filenames
    extracted_numbers = [extract_number(filename) for filename in sorted_filenames]
    print("len is : ", len(name_list))
    print("sorted_filenames len is : ", len(sorted_filenames))
    print("extracted_numbers len is : ", len(extracted_numbers))

    data_list = []
    for name in sorted_filenames:
        tmp_data = sp.loadmat(name)

        tmplist = []
        # if name == "F_amass_Subject_7.mat":
        #     IPython.embed()
        if tmp_data[list(tmp_data.keys())[3]]["move"][0][0].shape[0] == 22:
            print(name)
        # for i in range(tmp_data[list(tmp_data.keys())[3]]["move"][0][0].shape[0]):
        for i in range(21):
            # print("i is", i)
            # print(name)
            if not tmp_data[list(tmp_data.keys())[3]]["move"][0][0][i][0].size:
                tmplist.append(" ")
                continue
            if (
                tmp_data[list(tmp_data.keys())[3]]["move"][0][0][i][0][0][
                    "description"
                ][0][0]
                == "crossarms"
            ):
                tmplist.append("cross_arms")
            elif (
                tmp_data[list(tmp_data.keys())[3]]["move"][0][0][i][0][0][
                    "description"
                ][0][0]
                == "jumping_jack"
            ):
                tmplist.append("jumping_jacks")
            else:
                tmplist.append(
                    tmp_data[list(tmp_data.keys())[3]]["move"][0][0][i][0][0][
                        "description"
                    ][0][0]
                )
            ctr += 1
        data_list.append(tmplist)

    # with open("GFG", "w") as f:
    #     # using csv.writer method from CSV package
    #     write = csv.writer(f)

    #     # write.writerow(fields)
    #     write.writerows(data_list)

    data_array = np.array(data_list)
    print("valid action clips in total:", ctr + 1)
    np.save("movilabels.npy", data_array)

    # mat_contents1 = sp.loadmat("F_amass_Subject_15.mat")
    # mat_contents1["Subject_15_F_amass"]["move"][0][0][0][0][0]["description"][0][0]


if __name__ == "__main__":
    main()
