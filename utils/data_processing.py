from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

### DATA

transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                ])

class VLSP_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anno_id, image, question, answer = self.data.iloc[idx]
        image_path = "/kaggle/working/data/" + image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return anno_id, image, question, answer
    

df_train = pd.read_csv("/Users/duyhoang/Documents/Research/VQA/VQA_Vi/csv/ViTextVQA_train.csv")

train_vlsp_dataset = VLSP_Dataset(df_train, transform=transforms)
print(train_vlsp_dataset[0])
print(len(train_vlsp_dataset))