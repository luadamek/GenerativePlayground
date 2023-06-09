from generative_playground.config import WORKDIR
from generative_playground.dataset_loader import ImageLoader
import os
import glob
import pandas as pd

data_wildcard = os.path.join(WORKDIR, "datasets/unzipped/Humans/*.jpg")
files = glob.glob(data_wildcard)
loader = ImageLoader(files)

clean_files = []

frame_as_dict = {}
frame_as_dict["image_shape"] = []
frame_as_dict["last_image_dimension"] = []
frame_as_dict["shape_dim_zero"] = []
frame_as_dict["shape_dim_one"] = []
frame_as_dict["filename"] = []

print("Finding corrupted images")
for i in range(0, len(loader)):
    if i % 50 == 0:
        print(f"Done {i}")
    data, filename = loader[i]
    shape = data.shape
    frame_as_dict["image_shape"].append(len(shape))
    frame_as_dict["last_image_dimension"].append(shape[-1])
    shape_dim1 = -1
    if len(shape) > 0:
        shape_dim1 = shape[0]
    shape_dim2 = -2
    if len(shape) > 1:
        shape_dim2 = shape[1]
    frame_as_dict["shape_dim_zero"].append(shape_dim1)
    frame_as_dict["shape_dim_one"].append(shape_dim2)
    frame_as_dict["filename"].append(filename)


frame = pd.DataFrame.from_dict(frame_as_dict)

frame.to_csv(os.path.join(WORKDIR, "datasets/unzipped/metadata.csv"))
