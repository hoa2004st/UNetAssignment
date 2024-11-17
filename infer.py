import torch
from torchvision import transforms
from PIL import Image
import argparse
import segmentation_models_pytorch as smp

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to expected input size
    transforms.ToTensor(),  # Convert image to tensor
])

# Load model
def load_model(checkpoint_path):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=3     
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode
    return model

# Inference
def predict(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    # Assuming output is a segmented mask
    return output

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script for image segmentation')
    parser.add_argument('--image_path', type=str, help='Path to input image', required=True)
    args = parser.parse_args()

    # Load the model
    model = load_model('model.pth')

    # Predict
    output = predict(model, args.image_path)
    print(output.shape)

    # Save the result (convert tensor back to image and save)
    output_image = transforms.ToPILImage()(output.squeeze(0))
    output_image.save('output_image.png')
    print("Segmented image saved as output_image.png")
