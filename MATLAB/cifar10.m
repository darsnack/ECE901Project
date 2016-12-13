%% Build CNN model

% To do:
% create weights1 as a 3x3x3 array, set value, dim, in, out
% create bias1 as a 4x1 array, set value and dim
% create weights2 as 16x16x4 array, set value, dim, in, out
% create bias2 as 10x1 array, set value and dim

% Read sample as struct
% create layerIn, set fields value, height, width, depth
% create label vector

% CNN Parameters
kernelDim = 2;
kernelStride = 1;
poolDim = 2;
poolStride = 1;
channelsIn = 3;
channelsOut = 4;
numClasses = 10;

conv1 = conv_layer(layerIn,weights1,bias1);
pool1 = max_pool(conv1,poolDim,poolStride);
fc1 = fc_layer(pool1,weights2,bias2);

% update weights and biases

% report loss
loss = report_l2_loss(labels,prediction);

% fprintf('\n Step %d: loss = %8.4g ', ,loss);



% report overall training error at the end