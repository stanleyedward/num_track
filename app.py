import streamlit as st
from PIL import Image
import numpy as np
import torch
from torch.nn.functional import normalize
import torchvision
from torch import nn
from torchvision import transforms
from pathlib import Path
from functions import MNISTModelV1
from functions import make_predictions
from functions import make_predictions2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as fn


MODEL_SAVED_PATH = Path('models/mnist_tinyvgg_model.pth')


def main():
    st.title('tinyVGG Web Classifier')
    st.write('upload any image with single numbers digits to identify')

    file = st.file_uploader('Please upload an image', type=['jpg','png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width = True)
        transformation = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                  transforms.Resize((28,28)),
                                  transforms.PILToTensor()])
        transformed_image = transformation(image)
        test_image = transformed_image.unsqueeze(dim=0)
        st.write(test_image.shape)
        finaler_image = test_image.to(torch.float)
        normalizedd_tensor = normalize(finaler_image, p=2.0, dim=1)
        st.write(finaler_image.dtype)
        model = MNISTModelV1(input_shape=1,
                             hidden_units=10,
                             output_shape=10)
        model.load_state_dict(torch.load(f=MODEL_SAVED_PATH))
        st.write("model loaded")
        # st.write(model.state_dict())
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
        # #make preds
        prediction = make_predictions2(model=model,
                                       img_array=normalizedd_tensor)
        st.write(f'Prediction: {prediction}')
        # fig, ax = plt.subplots()
        # y_pos = np.arange(len(class_names))
        # ax.barh(y_pos, prediction, align='center')
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(class_names)
        # ax.invert_yaxis()
        # ax.set_xlabel("Probability")
        # ax.set_title("Number Predictions")

        # st.pyplot(fig)

    else:
        st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
    main()
