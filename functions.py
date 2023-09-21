import torch
from torch import nn

class MNISTModelV1(nn.Module):
    """
    TinyVGG model Arch.
    mddel from CNN explainers website https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, 
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            #create a conv layer
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1), #values wew can set ourselfs are called hyperparameters called nn conv2d
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)          
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=1),
                     
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,#there is a trick to calculate this!
                      out_features=output_shape),
            
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        # print("output shape fp conv layer 2",x.shape) #use this to get dim of in features in classification layer
        x = self.classifier(x)
        return x
    

def make_predictions(model: nn.Module,
                     img_array: torch.Tensor):
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
    model.eval()
    y_preds = []
    with torch.inference_mode():
        #forward pass
        output = model(img_array)
        pred_probs = torch.softmax(output, dim=1)
        y_pred = pred_probs.squeeze()
        numpy_preds = y_pred.numpy()
        
    return y_preds

def make_predictions2(model: nn.Module,
                     img_array: torch.Tensor):
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
    model.eval()
    with torch.inference_mode():
        #forward pass
        output = model(img_array)
        pred_probs = torch.softmax(output, dim=1)
        print(pred_probs.shape)
        pred_labels = torch.argmax(pred_probs, dim=1)

    return pred_labels.item()