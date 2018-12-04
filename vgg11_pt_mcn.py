
import torch
import torch.nn as nn


class Vgg11_pt_mcn(nn.Module):

    def __init__(self):
        super(Vgg11_pt_mcn, self).__init__()
        self.meta = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225],
                     'imageSize': [224, 224]}
        self.features_0 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_1 = nn.ReLU()
        self.features_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_3 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_4 = nn.ReLU()
        self.features_5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_6 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_7 = nn.ReLU()
        self.features_8 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_9 = nn.ReLU()
        self.features_10 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_11 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_12 = nn.ReLU()
        self.features_13 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_14 = nn.ReLU()
        self.features_15 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.features_16 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_17 = nn.ReLU()
        self.features_18 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self.features_19 = nn.ReLU()
        self.features_20 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.classifier_0 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.classifier_1 = nn.ReLU()
        self.classifier_2 = nn.Dropout(p=0.5)
        self.classifier_3 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.classifier_4 = nn.ReLU()
        self.classifier_5 = nn.Dropout(p=0.5)
        self.classifier_6 = nn.Linear(in_features=4096, out_features=1000, bias=True)

    def forward(self, data):
        features_0 = self.features_0(data)
        features_1 = self.features_1(features_0)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_4 = self.features_4(features_3)
        features_5 = self.features_5(features_4)
        features_6 = self.features_6(features_5)
        features_7 = self.features_7(features_6)
        features_8 = self.features_8(features_7)
        features_9 = self.features_9(features_8)
        features_10 = self.features_10(features_9)
        features_11 = self.features_11(features_10)
        features_12 = self.features_12(features_11)
        features_13 = self.features_13(features_12)
        features_14 = self.features_14(features_13)
        features_15 = self.features_15(features_14)
        features_16 = self.features_16(features_15)
        features_17 = self.features_17(features_16)
        features_18 = self.features_18(features_17)
        features_19 = self.features_19(features_18)
        features_20 = self.features_20(features_19)
        classifier_flatten = features_20.view(features_20.size(0), -1)
        classifier_0 = self.classifier_0(classifier_flatten)
        classifier_1 = self.classifier_1(classifier_0)
        classifier_2 = self.classifier_2(classifier_1)
        classifier_3 = self.classifier_3(classifier_2)
        classifier_4 = self.classifier_4(classifier_3)
        classifier_5 = self.classifier_5(classifier_4)
        classifier_6 = self.classifier_6(classifier_5)
        return classifier_6

def vgg11_pt_mcn(weights_path=None, **kwargs):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = Vgg11_pt_mcn()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
