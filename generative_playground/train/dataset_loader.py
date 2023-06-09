import matplotlib.pyplot as plt

class ImageLoader:
    def __init__(self, file_list):
        self.file_list = [el for el in file_list]
    
    def __getitem__(self, i):
        return plt.imread(self.file_list[i]), self.file_list[i]
    
    def __len__(self):
        return len(self.file_list)
