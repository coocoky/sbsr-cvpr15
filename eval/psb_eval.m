function psb_eval(test_case)

disp = 1;

cache_dir = './cache';
data_dir = './data';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluation setup

evalrst_dir = 'PSB_Evaluation_Ours';
testname = test_case.test_mode; % 'train', 'test', 'all'
vp_num = test_case.vp_num;
name = test_case.name;
suffix = test_case.suffix;
epoch_list = test_case.epoch_list;

switch testname
  case 'train',
    sketch_data = load(sprintf('%s/psb_sketch_train.mat', data_dir));
    view_data = load(sprintf('%s/psb_view_train.mat', data_dir));
  case 'test',
    sketch_data = load(sprintf('%s/psb_sketch_test.mat', data_dir));
    view_data = load(sprintf('%s/psb_view_test.mat', data_dir));
  otherwise,
    error(['Unknown testname: ', testname]);
end

sketches = sketch_data.datax;
sketch_labels = sketch_data.datay;
views = view_data.datax;
view_labels = view_data.datay;
% view_flags  = view_data.data_valid;

number_of_queries=size(sketches, 1);
sketch_num = size(sketches, 1);
view_num = size(views, 1);

linestyle_epoch = {'-*b', '-*r', '-*m', '-*c', '-+b', '-*k', '-ob', '-or', '-om', '-oc', '-ok', '-^b'};
plotentries = cell(3, 0);
alg_names = {};

PrecisionAll = [];

for ii = 1 : length(epoch_list)
  epoch = epoch_list(ii);
  
  evalrank_dir = fullfile(evalrst_dir, [test_case.case_name, '-epoch', num2str(epoch)]);
  if ~exist(evalrank_dir, 'dir')
    mkdir(evalrank_dir);
  end
  
  test_feat_file = sprintf('%s/feats-%s-%s/%s-%s-epoch%d-%s.mat', cache_dir, name, suffix, name, suffix, epoch, testname);
  if ~exist(test_feat_file, 'file')
    fprintf('feature file does not exist: %s.\n', test_feat_file);
    continue;
  end
  
  fprintf('Loading feature distance ...\n');
  load(test_feat_file, 'sketch_feat', 'view_feat', 'dist_mat');
  dist = dist_mat;
  
  % fold score matrix according to view labels
  model_num = view_num / vp_num;
  vds = zeros(sketch_num, model_num, vp_num);
  for i = 1 : vp_num
    vds(:, :, i) = dist(:, model_num*(i-1)+1:model_num*i);
  end
  [dist, dist_ind] = min(vds, [], 3);
  [scores, orders] = sort(dist, 2);
  
  fprintf('Performing evaluation ...\n');
  results = zeros(size(dist));
  for qqq = 1 : number_of_queries
    results(qqq, :) = view_labels(orders(qqq, :));
  end
  [Precision, Recall] = SHREC_2013_eval_quick(evalrank_dir, testname, sketch_labels, results, 1);
  fprintf('first precision (recall=0.05) = %f \n', Precision(1));
  
  if (isempty(PrecisionAll))
    PrecisionAll = Precision;
  else
    PrecisionAll = [PrecisionAll; Precision];
  end
  
  plotentries(:, end+1) = {Recall, Precision, linestyle_epoch{ii}};
  alg_names(end+1) = {['Ours-epoch', num2str(epoch)]};
end

hf = figure(1);
% if ~disp, set(hf, 'Visible', 'off'); end
clf();
plot(plotentries{:});
hl = legend(alg_names);
set(hl,'Interpreter','none', 'Fontsize', 9)
xlabel('Recall', 'Fontsize', 12);
ylabel('Precision', 'Fontsize', 12)
ylim([0,0.75]);
% set(gcf, 'Units','centimeters', 'Position',[78 22 12 10]);
set(gcf, 'PaperPositionMode','auto')
[~, name, ~] = fileparts(test_feat_file);
figure_file = sprintf('%s/%s.png', evalrst_dir, name);
fprintf('saving figure as: %s.\n', figure_file);
print(sprintf('-f%d',hf),'-dpng', figure_file );




