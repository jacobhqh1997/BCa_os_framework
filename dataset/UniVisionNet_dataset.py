
class UniVisionNet(Dataset):

    def __init__(self, h5_paths, transform_seg=None):
        super(percevierDatasetLoader, self).__init__()

        self.images_filepaths = h5_paths
        self.transform_seg = transform_seg

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        ########################patches#########################
        h5_path = self.images_filepaths[idx]
        hospital = '1st_train'
        base_dir = INFO_PATH
        self.data = pd.read_csv(base_dir+hospital+'.csv')
        
        data = torch.load(h5_path)
        features = data['features']
        coords = data['coords']
        ID = h5_path.split('/')[-1][:-12]  # Adjust the slicing if necessary
        pd_index = self.data[self.data['WSIs'].isin([ID])].index.values[0]
        T = self.data['death_time'][pd_index]
        O = self.data['death_status'][pd_index]
        ########################segmaps#########################
        base_seg_path = HEATMAP_PATH
        seg_filepath = base_seg_path + h5_path.split('/')[-1][:-12] + '.npy'
        seg = np.load(seg_filepath)
        if self.transform_seg is not None:
            seg = self.transform_seg(image=seg)["image"]
        return features, seg, T, O, h5_path, seg_filepath
