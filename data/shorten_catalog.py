import json

with open("data/s25.json", "r") as f:
    s25 = json.load(f)

classes = []
for number, info in s25["classes"].items():
    classes.append(f"{number} {info['name']}")

with open("data/s25_names.txt", "a") as f:
    class_string = ", ".join(classes)
    class_string = class_string[0:len(class_string) - 2]
    f.write(class_string)