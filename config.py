import torch

'''
class DefaultConfigs(object):
    train_data = "./human_protein/train/" # where is your train data
    test_data = "./human_protein/test/"   # your test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/best_models/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 50
'''

class DefaultConfigs(object):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dirs = ''
    test_dirs = ''
    weight_dirs = ''
    num_classes = 28
    channels = 4
    img_weight = 512
    img_height = 512
    epochs = 5
    batch_size = 2
    lr = 0.003

config = DefaultConfigs()
