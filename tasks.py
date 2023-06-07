from invoke import task

@task
def download_data(cmd):
    cmd.run("python -m scripts.download_data")
