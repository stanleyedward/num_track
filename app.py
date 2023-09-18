import streamlit as st
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from pathlib import Path
from model_arch import MNISTModelV1
from model_arch import make_predictions
import matplotlib.pyplot as plt
import torchvision.transforms.functional as fn
import cv2

MODEL_SAVED_PATH = Path('models/mnist_tinyvgg_model.pth')

def main():
    st.title('tinyVGG Web Classifier')
    st.write('upload any image with single numbers digits to identify')

    file = st.file_uploader('Please upload an image', type=['jpg','png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width = True)
        # resized_image = fn.resize(image, size=[28,28])
        # convert_tensor = transforms.ToTensor()
        # resized_image = image.resize((28,28))
        # img_array = np.array(resized_image) / 255
        # img_array = img_array.reshape((1,1,28,28))
        # img_array = convert_tensor(resized_image)
        # resized_image = convert_tensor(resized_image)
        # img_array = resized_image.unsqueeze(dim=0)
        image_8 = cv2.imread(str(file))
        # following line may or may not be necessary
        image_8 = cv2.cvtColor(image_8, cv2.COLOR_BGR2GRAY)

        # you can divide numpy arrays by a constant natively
        image_8 /= 255.

        # This makes a 4d tensor (batched image) with shape [1, channels, width, height]
        tensor_8 = torch.Tensor(image_8).unsqueeze(dim=0)


        model = MNISTModelV1(input_shape=1,
                             hidden_units=10,
                             output_shape=10)
        model.load_state_dict(torch.load(f=MODEL_SAVED_PATH))

        class_names = ['0 - zero',
                '1 - one',
                '2 - two',
                '3 - three',
                '4 - four',
                '5 - five',
                '6 - six',
                '7 - seven',
                '8 - eight',
                '9 - nine']
        #make preds
        predictions = make_predictions(model=model,
                                       img_array=tensor_8)
        fig, ax = plt.subplots()
        y_pos = np.arange(len(class_names))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title("Number Predictions")

        st.pyplot(fig)

    else:
        st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
    main()