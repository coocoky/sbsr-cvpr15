
data_dir = './data';

ori_filename = sprintf('%s/sketch_train.mat', data_dir);
[pa, name, ext] = fileparts(ori_filename);

warp_filename = sprintf('%s/%s_augment_warp.mat', pa, name);
gen_data_augmentation(ori_filename, warp_filename)
warp_filename = sprintf('%s/%s_augment_warp2.mat', pa, name);
gen_data_augmentation(ori_filename, warp_filename)

