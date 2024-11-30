import network
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms

# Assuming you have a folder of images
image_folder = '/Users/kechenhua/Desktop/images'  # Replace with the path to your images
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

# Load and preprocess the images
images = []
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')  # Open image and ensure it's in RGB format
    images.append(image)
# Define the transformation (convert PIL.Image to Tensor)
transform = transforms.Compose([
    transforms.Resize(512),  # Increase input image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Apply the transformation to each image and create a list of tensors
image_tensors = [transform(img) for img in images]
# Stack the list of images into a single tensor (batch of images)
images = torch.stack(image_tensors)  # Shape: (N, C, H, W), where N is the batch size



#Load the pretrained model:
model = network.modeling.__dict__['deeplabv3_resnet50'](num_classes=21, output_stride=8)
model.load_state_dict( torch.load( '/Users/kechenhua/Desktop/computer-vision-final-project-six/deeplabv3/best_deeplabv3_resnet50_voc_os16.pth', map_location=torch.device('cpu') )['model_state']  )

#Visualize segmentation outputs:
outputs = model(images)
preds = outputs.max(1)[1].detach().cpu().numpy()
colorized_preds = val_dst.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
# Do whatever you like here with the colorized segmentation maps
colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image


# Optionally, visualize using matplotlib
plt.imshow(colorized_preds[0])  # Show the first image in the batch
plt.axis('off')  # Hide axes
plt.show()

# Optionally, save the colorized output
# colorized_preds_image.save('segmentation_output.png')