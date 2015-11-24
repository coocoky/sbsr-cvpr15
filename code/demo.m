addpath('./eval');

config_file = './config/exp-shrec.cfg';
test_cases = parseConfig(config_file);
shrec13_eval(test_cases{1});

% config_file = './config/exp-shrec-single.cfg';
% test_cases = parseConfig(config_file);
% shrec13_eval(test_cases{1});

% config_file = './config/exp-shrec14.cfg';
% test_cases = parseConfig(config_file);
% shrec14_eval(test_cases{1});

% config_file = './config/exp-psb.cfg';
% test_cases = parseConfig(config_file);
% psb_eval(test_cases{1});
