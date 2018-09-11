import subprocess


if __name__ == '__main__':
    subprocess.run(["git", "pull", "origin", "master"])
    subprocess.run(["python3", "server.py"])
