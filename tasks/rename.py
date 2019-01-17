import os
import re

template_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "templates", "hadoken_templates")

if __name__ == '__main__':

    rule = re.compile(r'\d+\.5\..+')

    for file in os.listdir(template_dir):
        if rule.match(file):
            new_name = file.replace(".5.", ".8.")
            if os.path.exists(os.path.join(template_dir, new_name)):
                new_name = "10" + new_name
            print(file, new_name)
            os.rename(os.path.join(template_dir, file), os.path.join(template_dir, new_name))
