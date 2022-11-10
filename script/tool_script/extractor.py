import json


path = "/data_path/night_test_qa.txt"

with open (path) as input_file:
    json_paths = [_.strip() for _ in input_file]
    image_jsons = []
    for json_path in json_paths:
        with open (json_path) as json_obj:
            relative_datas = json.load(json_obj).get("relative_images_data", None)
            
            for relative_data in relative_datas:
                camera_orientation = relative_data.get("camera_orientation", None)
                if camera_orientation in ["front_right_camera",  "rear_right_camera", "rear_left_camera", "front_left_camera"]:
                    image_json = relative_data.get("image_json", None)
                    if image_json != "":
                        image_jsons.append("/" + image_json)
                    
target_path = "/data_path/night_test_qa_frame.txt"
with open (target_path, "w") as output_file:
    for image_json in image_jsons:
        output_file.writelines(image_json + "\n")
                
print(target_path)