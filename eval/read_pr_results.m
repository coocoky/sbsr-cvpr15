function [Precision, Recall] = read_pr_results(filename)

fin = fopen(filename, 'r');
if fin < 0
  error(['can not open file ', filename]);
end

cnt = 0;
Precision = [];
Recall = [];

while 1
  line = fgets(fin);
  if (line == -1), break; end
  if line(1) < '0' || line(1) > '9', continue; end
  
  [A, n] = sscanf(line, '%f %f', [1, 2]);
%   if n < 2, break; end
  cnt = cnt + 1;
  Precision(cnt) = A(2);
  Recall(cnt) = A(1);
end

fclose(fin);