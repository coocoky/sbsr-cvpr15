function [Pmat_ori, Rmat_ori, names, plotentries] = read_eval_results(evalrst_dir, testname, disp_flag)
if nargin < 2
  disp_flag = 1;
end

cnt = 0;
Pmat = cell(0);
Rmat = cell(0);
names = cell(0);

subfolders = dir(evalrst_dir);
for i = 1 : length(subfolders)
  if ~subfolders(i).isdir || subfolders(i).name(1) == '.' 
    continue;
  end
  
  cnt = cnt + 1;
  names{cnt} = subfolders(i).name;
  rst_file = dir(sprintf('%s/%s/*PR_%s.txt', evalrst_dir, subfolders(i).name, testname));
  assert(length(rst_file)==1);
  rst_filename = fullfile(evalrst_dir, subfolders(i).name, rst_file(1).name);
  [Pmat{cnt}, Rmat{cnt}] = read_pr_results(rst_filename);
end

Pmat_ori  = cell2mat(Pmat')';
Rmat_ori = cell2mat(Rmat')';

[~, order] = sort(Pmat_ori(1, :), 'descend');
Pmat = Pmat_ori(:, order);
Rmat = Rmat_ori(:, order);
names = names(order);

markstyle = {'o', 'o', 'o', '+', 'o', '+', '+', 'o', 'o', 'p', 'x', '^', 's', 's', 's', 'v', '<', '<'};
plotentries = cell(3, length(order));

for i = 1 : length(names)
  plotentries(:, i) = {Rmat(:,i), Pmat(:,i), ['-',markstyle{i}]};
end

if disp_flag
  clf();
  plot(plotentries{:});
  h = legend(names);
  set(h,'Interpreter','none')
end
