import os
import sys
import numpy as np
import torch
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import MyDataset

device = torch.device("cuda")
model = torch.load("./bestmodel.pkl?dl=1")
model.to(device)
model.eval()

transform = transforms.ToTensor()

def gray2rgb(gray_img):
    rgb_img = np.concatenate((
        gray_img[:,:,np.newaxis],
        gray_img[:,:,np.newaxis],
        gray_img[:,:,np.newaxis],
    ), axis=2)
    
    return rgb_img

def predict(x):
    # x is an rgb image, so it needs to be converted into grayscale image
    # x.shape = (batch_size=1, 48, 48, 3)
    with torch.no_grad():
        x_gray = x[0,:,:,0]
        x_gray = transform(x_gray).unsqueeze(0)
        x_gray = x_gray.to(device)

        output = model(x_gray).cpu().numpy()

    # output.shape = (batch_size=1, 7)
    return output

def segmentation(x):
    return slic(x)


if __name__ == "__main__":
    classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
 
    dataset = MyDataset.ImageDataset(sys.argv[1], transform=None)
    data = [
        dataset[374],        # 0: angry
        dataset[2275],       # 1: disgust
        dataset[84],         # 2: fear
        dataset[7],          # 3: happy
        dataset[70],         # 4: sad
        dataset[194],        # 5: surprise
        dataset[217]         # 6: neutral
    ]
    
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2], exist_ok=True)

    for i in range(len(data)):
        x, y = data[i]
        x_rgb = gray2rgb(x)
        
        explainer = lime_image.LimeImageExplainer(random_state=69)
        
        explaination = explainer.explain_instance(
                                    image=x_rgb,
                                    classifier_fn=predict,
                                    batch_size=1,
                                    segmentation_fn=segmentation
                                )
        
        image, mask = explaination.get_image_and_mask(
                                    label=y.item(),
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )

        # save the image
        plt.figure(i)
        plt.imshow(image)
        plt.savefig(os.path.join(sys.argv[2], "fig3_{}.jpg".format(i)))
        plt.close("all")

        print("Finish saving class {}: {}.".format(i, classes[i]))