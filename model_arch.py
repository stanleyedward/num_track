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
                     img_array):
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
        X = torch.Tensor(img_array)
        #forward pass
        y_logit = model(X).squeeze() #raw output of a model with linear layer at the end
        #turn logits to pred probs to labels
        y_pred_prob = torch.softmax(y_logit, dim=0)
        y_pred_label = y_pred_prob.argmax(dim=1)
        #put preds on CPU for eval
        y_preds.append(y_pred_label)

    return y_preds