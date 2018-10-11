import os
import shutil

pwd = os.path.dirname(os.path.realpath(__file__))
hierarchy_path = os.path.join(pwd, "..", "..", "new_templates")
target_path = os.path.join(pwd, "..", "..", "large_template_front")
if not os.path.exists(target_path):
    os.makedirs(target_path)

if __name__ == '__main__':
    print(hierarchy_path)
    for subdir in os.listdir(hierarchy_path):
        full_subdir = os.path.join(hierarchy_path, subdir)
        print(full_subdir)
        if os.path.isdir(full_subdir):
            counter = 0
            label = str(abs(int(subdir)))
            for filename in os.listdir(full_subdir):
                pwf = os.path.join(full_subdir, filename)
                if os.path.isfile(pwf):
                    new_name = str(counter) + "." + str(label) + "." + str(0 if int(subdir) > 0 else -1) + ",0,100,100.png"
                    shutil.copy(pwf, os.path.join(target_path, new_name))
                    counter = counter + 1
