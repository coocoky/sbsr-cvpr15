function gen_data_augmentation(ori_filename, warp_filename)

load(ori_filename, 'datax', 'datay', 'data_imid');

N = size(datax, 1);
w = 100;
h = 100;

datax_aug = zeros(N, w*h);
sx = zeros(1, N);
sy = zeros(1, N);
a = zeros(1, N);
tx = 0.5;  ty = 0.5;

for i = 1:N
  patch = reshape(datax(i,:), w, h)';
  sxi = rand(1)*0.15 + 0.90;  % scale range: [0.90 ~ 1.05]
  syi = rand(1)*0.15 + 0.90;  % scale range: [0.90 ~ 1.05]
  ai  = (rand(1)*6 - 3)*pi/180; % rotate range: [-3 ~ 3]
  
  sx(i) = sxi;  sy(i) = syi; a(i) = ai;
  
  % make transformation matrix (T)
  T  = [ sxi*cos(ai) sxi*sin(ai) tx ; -syi*sin(ai) syi*cos(ai) ty ; 0 0 1 ];
  [Xp, Yp] = ndgrid([1:w], [1:h]);
  wp = w; hp = h; n = wp*hp;
  X = T \ [ Xp(:) Yp(:) ones(n,1) ]';  % warp
  
  % re-sample pixel values with bilinear interpolation
  xI = reshape( X(1,:),wp,hp)';
  yI = reshape( X(2,:),wp,hp)';
  patch_new = interp2(patch(:,:), xI, yI, 'cubic');
  
  datax_aug(i, :) = reshape(patch_new', [1, 10000]);
end

datax = datax_aug;
datax(isnan(datax)) = 0;
save(warp_filename, 'datax', 'datay', 'data_imid', 'sx', 'sy', 'a');

