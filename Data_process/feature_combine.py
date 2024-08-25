import os
import torch


feature_dir = 'path/to/giga'
coord_dir = 'path/to/Macro_feature'
save_dir = 'path/to/giga_combined'

os.makedirs(save_dir, exist_ok=True)


feature_files = [f for f in os.listdir(feature_dir) if f.endswith('_features.pt')]

for feature_file in feature_files:
  
    base_name = feature_file.replace('_features.pt', '.pt')
    coord_file = os.path.join(coord_dir, base_name)
    

    if not os.path.exists(coord_file):
        print(f"skip：{coord_file}")
        continue  
    
  
    feature_path = os.path.join(feature_dir, feature_file)
    features_data = torch.load(feature_path)
    features = features_data['features']
    coords = features_data['coords']
    
   
    coord_data = torch.load(coord_file)
    coord_features = coord_data.repeat(features.size(0), 1) 
    coord_features = coord_features.to(features.device)
    
   
    combined_features = torch.cat((features, coord_features), dim=1)
    features_data['features'] = combined_features
    

    save_path = os.path.join(save_dir, feature_file)
    torch.save(features_data, save_path)

