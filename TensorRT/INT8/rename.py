# import os 

# folder_path = "raw_verify/out_data_6"

# folder = os.listdir(folder_path)

# list_json = [file for file in folder if file.endswith(".json")]
# list_image= [file for file in folder if (file.endswith(".jpg") or file.endswith(".png"))]

# for idx, name in enumerate(list_json):
#     old_json = folder_path + "/" + name
#     new_json = folder_path + "/" + f"{idx + 1:06}.json"
    
#     old_image = folder_path + "/" + name[:-5] + ".jpg"
#     new_image = folder_path + "/" + f"{idx + 1:06}.jpg"
    
#     os.rename(old_json , new_json)
#     os.rename(old_image , new_image)
    



import os 

# for m in os.listdir("D:/Download/Test_Chau/Test_Chau"):

folder_path = "data_calib"

folder = os.listdir(folder_path)

list_image= [file for file in folder if (file.endswith(".jpg") or file.endswith(".png"))]

for idx, name in enumerate(list_image):

    old_image = folder_path + "/" + name[:-4] + ".jpg"
    tail = ".jpg"
    if not os.path.exists(old_image):
        old_image = folder_path + "/" + name[:-4] + ".png"
        tail = ".png"
    new_image = folder_path + "/" + f"{idx + 1:06}" + tail
    
    os.rename(old_image , new_image)
    
