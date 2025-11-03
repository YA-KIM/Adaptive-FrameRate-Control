import os
import torchvision.transforms as transforms
import torchvision


def parse_log_line(self, log_line):
        """
        Parse a single log line to extract the relevant information.
        """
        try:
            parts = log_line.strip().split(", ")
            data = {}

            for part in parts:
                key, value = part.split(": ")
                key = key.strip()
                value = value.strip()

                if key == "BBox":
                    data[key] = list(map(float, value[1:-1].split(", ")))
                elif key in ["Vel", "Acc"]:
                    data[key] = list(map(float, value[1:-1].split(", ")))
                elif key == "AngVel":
                    data[key] = float(value)
                else:
                    data[key] = value

            return data
        except Exception as e:
            print(f"Error parsing log line: {log_line}\n{e}")
            return None


def read_voc_dataset(path, year, download):
    T = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            ])
    voc_data =  torchvision.datasets.VOCDetection(path, year=year, image_set='train', transform=T, download=download)
    voc_val =  torchvision.datasets.VOCDetection(path, year=year, image_set='val', transform=T, download=download)

    return voc_data, voc_val


#First of all, we can try multi processing betweeb Detact_Track,,, Training Code should be modified
# of course, Training Data should be prepared,.

