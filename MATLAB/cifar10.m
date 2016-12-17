%% Training CNN Model on CIFAR-10 Dataset

clear;
close all;
clc;

% CNN Parameters
imageSize = 32;
kernelDim = 3;
kernelStride = 1;
poolDim = 2;
poolStride = 1;
channelsIn = 3;
channelsOut = 4;
numClasses = 10;
learningRate = 0.001;
info = struct('iter',0,'loss',0);

% Initialize weights and biases
weights1 = struct('dim',kernelDim,'in',channelsIn,'out',channelsOut,'value',[]);
weights1.value = 0.05*randn(weights1.dim,weights1.dim,weights1.in,weights1.out);
% weights1.value = 0.125*ones(weights1.dim,weights1.dim,weights1.in,weights1.out);

bias1 = struct('dim',channelsOut,'value',[]);
bias1.value = zeros(bias1.dim,1);

weights2 = struct('dim',imageSize/poolDim,'in',channelsOut,'out',numClasses,'value',[]);
weights2.value = 0.04*randn(weights2.out,weights2.dim*weights2.dim*weights2.in);
% weights2.value = 0.125*ones(weights2.out,weights2.dim*weights2.dim*weights2.in);

bias2 = struct('dim',numClasses,'value',[]);
bias2.value = zeros(bias2.dim,1);

lossVec = [];

numBatches = 2;
batchSize = 10000;

for i = 1:numBatches
    [data,classes] = read_cifar_data(i);  
    data = data/255;    % divide pixel values by 255
    sampleOrder = randperm(batchSize);
    for j = 1:batchSize
        
        % Initialize input layer 
        img = reshape(data(sampleOrder(j),:),[imageSize,imageSize,channelsIn]);
        img = transpose_layer(img,channelsIn);
        for ii = 1:channelsIn
           singleImg = img(:,:,i);
           singleImg = singleImg(:);
           adjStdDev = max(std(singleImg),1/imageSize);
           singleImg = (singleImg - mean(singleImg))/adjStdDev;
           img(:,:,i) = reshape(singleImg,[imageSize,imageSize]);
        end
        layerIn = struct('height',imageSize,'width',imageSize,'depth',channelsIn,'value',img,'derivative',[]);
                
        % create label vector
        label = zeros(numClasses,1);
        label(classes(j)+1) = 1;
        
        conv1 = conv_layer(layerIn,weights1,bias1);
        pool1 = max_pool(conv1,poolDim,poolStride);
        fc1 = fc_layer(pool1,weights2,bias2);
        
        % report loss
        info.loss = report_l2_loss(label,fc1.value);
        if (mod(info.iter,10)==0) 
            lossVec = [lossVec info.loss];
        end
        fprintf('\n Step %d: loss = %.4f ',info.iter,info.loss);
        
        % backpropagate errors     
        error1 = (fc1.value - label).*(fc1.derivative);
        error2 = fc_layer_propagate(error1,pool1,weights2);
        error3 = conv_layer_propagate(error2,conv1,poolDim);
        
        % update weights and biases
        [weights2,bias2] = fc_layer_update(weights2,bias2,error1,pool1,learningRate);
        [weights1,bias1] = conv_layer_update(weights1,bias1,error3,layerIn,learningRate);
        
        info.iter = info.iter + 1;
        
    end
end

% Plot loss every 100 samples
plot(lossVec(2:end));
xlabel('iterations (x100)');
ylabel('loss')

% report overall training error at the end