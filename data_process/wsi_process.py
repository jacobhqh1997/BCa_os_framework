def find_patches_from_slide(slide_path,):
    with openslide.open_slide(slide_path) as slide:
        width, height = dzg.level_dimensions[-2]
        thumbnail = slide.get_thumbnail((width//128, height//128))
    thumbnail = np.array(thumbnail)
    thumbnail_hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    h_channel = thumbnail_hsv[:, :, 0]
    thresh_h = threshold_otsu(h_channel)
    binary_h = h_channel < thresh_h
    patches = pd.DataFrame(pd.DataFrame(binary_h).stack())
    patches['is_tissue'] = ~patches[0]
    patches.drop(0, axis=1, inplace=True)
    patches['slide_path'] = slide_path
    samples = patches
    samples = samples[samples['is_tissue']] 
    samples = samples.copy()
    samples['tile_loc'] = list(samples.index)
    samples.reset_index(inplace=True, drop=True)
    return samples 
if __name__ == '__main__':
  slide_path = 'path/to/your/slide/'
  slide = openslide.OpenSlide(slide_path)
  tiles = DeepZoomGenerator(slide, tile_size=128, overlap=64, limit_bounds=False)
  size = (int((tiles.level_dimensions)[-2][1]/128),int((tiles.level_dimensions)[-2][0]/128))
  all_tissue_samples = find_patches_from_slide(slide_path)
