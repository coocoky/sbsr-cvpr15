function shrec14_eval(test_case)

disp = 1;

cache_dir = './cache';
data_dir = './data';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evaluation setup

evalrst_dir = 'SHREC14_Evaluation_Ours';
shrec14_evalrst_dir = 'SHREC14_Evaluation_Results';
testname = test_case.test_mode; % 'train', 'test', 'all'
vp_num = test_case.vp_num;
name = test_case.name;
suffix = test_case.suffix;
epoch_list = test_case.epoch_list;

switch testname
  case 'train',
    sketch_data = load(sprintf('%s/shrec14_sketch_train.mat', data_dir));
  case 'test',
    sketch_data = load(sprintf('%s/shrec14_sketch_test.mat', data_dir));
  case 'complete',
    sketch_data = load(sprintf('%s/shrec14_sketch_train.mat', data_dir));
    sketch_test_data = load(sprintf('%s/shrec14_sketch_test.mat', data_dir));
    sketch_data.datax = [sketch_data.datax; sketch_test_data.datax];
    sketch_data.datay = [sketch_data.datay; sketch_test_data.datay];
  otherwise,
    error(['Unknown testname: ', testname]);
end
view_data = load(sprintf('%s/shrec14_view_all.mat', data_dir));

sketches = sketch_data.datax;
sketch_labels = sketch_data.datay;
views = view_data.datax;
view_labels = view_data.datay;

number_of_queries=size(sketches, 1);
sketch_num = size(sketches, 1);
view_num = size(views, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do evaluation

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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% draw figure
[Pmat, Rmat, names_ours, plotentries_ours] = read_eval_results(evalrst_dir, testname, 0);
[Pmat_shrec, Rmat_shrec, names_shrec, plotentries_shrec] = read_eval_results(shrec14_evalrst_dir, testname, 0);

alg_names = [names_ours, names_shrec];
plotentries = [plotentries_ours, plotentries_shrec];

colorset = jet(size(plotentries, 2));
colorset = colorset(end:-1:1, :);
% colorset = colorset(2:end-1, :);
markerset = {'o','s','>','p','^'};

hf = figure(1);
if ~disp, set(hf, 'Visible', 'off'); end
clf();
set(0, 'DefaultAxesColorOrder', colorset);
hold on;
for i = 1 : size(plotentries, 2)
  marker = markerset{mod(i-1, length(markerset))+1};
  if strcmp(alg_names{i}(1:4), 'Ours') == 1,
    marker = '*';
  end
  plot(plotentries{1,i}, plotentries{2,i}, ['-', marker], ...
    'MarkerSize', 5, ...
    'Color', colorset(i, :));
end
hold off;

hl = legend(alg_names);
set(gca, 'Position', [0.1, 0.1, 0.85, 0.85], 'Fontsize', 12);
box on;
set(hl,'Interpreter','none', 'Fontsize', 9, 'Location', 'NorthEast')

xlabel('Recall', 'Fontsize', 12);
ylabel('Precision', 'Fontsize', 12)
ylim([0,0.65]);
%   set(hf, 'Position', [2800 120  960 800]);
set(gcf, 'Units','centimeters', 'Position',[4 9 13.5 14]);
set(gcf, 'PaperPositionMode','auto')
[~, name, ~] = fileparts(test_feat_file);
figure_file = sprintf('%s/%s.png', evalrst_dir, name);
fprintf('saving figure as: %s.\n', figure_file);
print(sprintf('-f%d',hf),'-dpng', figure_file );

