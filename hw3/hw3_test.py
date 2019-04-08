import sys
import csv
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import MyDataset

if __name__ == "__main__":
    device = torch.device("cuda")

    test_dataset = MyDataset.ImageDataset(file_path=sys.argv[1], transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    model = torch.load("./bestmodel.pkl?dl=1")
    model.to(device)
    model.eval()

    with torch.no_grad():
        with open(sys.argv[2], 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','label'])

            for i, data in enumerate(test_loader):

                inputs, img_id = data
                inputs = inputs.to(device)
                
                outputs = model(inputs)
                predict = torch.max(outputs, 1)[1].cpu().numpy()
                img_id = img_id.numpy()
                
                for j in range(len(img_id)):
                    writer.writerow([img_id[j], predict[j]])

            
        