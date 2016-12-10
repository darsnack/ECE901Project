% A function to read whatever batch input of data you want from CIFAR-10
% cifar-10-batches-mat folder must be in the root directory of the repo
% MATLAB code must be in the MATLAB directory of the repo
function [data, labels] = read_cifar_data(batch_num)
    filename = '../cifar-10-batches-mat/data_batch_';
    filename = [filename num2str(batch_num)];
    filename = [filename '.mat'];
    
    load(filename);
end