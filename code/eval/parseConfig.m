function test_cases = parseConfig(config_file)
% parse test case configuration file in following format
%
% [case_name]
% name: shrec
% suffix: default-suffix
% epoch_list: 10 20
% test_mode: test
% vp_num: 2

fid = fopen(config_file);
if fid<0, error('cannot open file %s\n',config_file); end;

test_cases = {};
test_case = [];
cnt = 0;
while ~feof(fid)
  line = strtrim(fgetl(fid));
  if isempty(line) || all(isspace(line)) || strncmp(line,'#',1) || strncmp(line,';',1),
    continue;
  end
  
  [key, val] = strtok(line,' ');
  
  if strncmp(key, '[', 1),
    if cnt > 0
      test_cases{end+1} = test_case;
    end
    
    case_name = key(2:end-1);
    test_case = struct('case_name', case_name, 'name', [], 'suffix', key(2:end-1), 'epoch_list', [], ...
                       'test_mode', 'test', 'vp_num', 2);
    cnt = cnt + 1;
  elseif key(end) == ':',
    key = key(1:end-1);
    val = strtrim(val);
    %       fprintf('%s=''%s''; \n',key,val);
    if strcmp(key, 'epoch_list')
      epoch_list = cellfun(@(x) int32(str2double(x)), strsplit(val, ' '));
      test_case.epoch_list = epoch_list;
    elseif strcmp(key, 'test_mode')
      test_case.test_mode = val;
    elseif strcmp(key, 'vp_num')
      test_case.vp_num = int32(str2double(val));
    else
      test_case = setfield(test_case, key, val);
    end
    
  end
  
end

if cnt > 0
  test_cases{end+1} = test_case;
end

fclose(fid);

