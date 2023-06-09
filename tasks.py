from invoke import task

@task
def download_data(cmd):
    cmd.run("python -m scripts.download_data")

@task
def create_clean_data_list(cmd):
    cmd.run("python -m scripts.generate_clean_data_list")
