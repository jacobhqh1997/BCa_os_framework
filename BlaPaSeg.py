from torchvision import datasets, models

def initialize_model(num_classes, feature_extract):
    model_ft = models.resnext50_32x4d(pretrained=False) 
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    input_size = 150
    return model_ft, input_size

num_classes = 8
feature_extract = False
model, input_size = initialize_model(num_classes, feature_extract)
BlaPaSeg = model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(model_dir)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
