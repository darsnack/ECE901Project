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
learningRate = 0.003;

info = struct('iter',0,'loss',0);

% Initialize weights and biases
weights1 = struct('dim',kernelDim,'in',channelsIn,'out',channelsOut,'value',[]);
weights1.value = (1/imageSize^2)*randn(weights1.dim,weights1.dim,weights1.in,weights1.out);
weights1.value = stochastic_quantize(weights1.value);
% weights1.value = 0.125*ones(weights1.dim,weights1.dim,weights1.in,weights1.out);

bias1 = struct('dim',channelsOut,'value',[]);
bias1.value = zeros(bias1.dim,1);

weights2 = struct('dim',imageSize/poolDim,'in',channelsOut,'out',numClasses,'value',[]);
weights2.value = (1/imageSize^2)*randn(weights2.out,weights2.dim*weights2.dim*weights2.in);
weights2.value = stochastic_quantize(weights2.value);
% weights2.value = 0.125*ones(weights2.out,weights2.dim*weights2.dim*weights2.in);

bias2 = struct('dim',numClasses,'value',[]);
bias2.value = zeros(bias2.dim,1);

numBatches = 2; % # of .mat data files to be read in
batchSize = 10000; % of training examples to use per batch
miniBatchSize = 128; % of of training examples in each mini batch
numMiniBatches = ceil(batchSize/miniBatchSize);
lastMiniBatchSize = mod(batchSize,miniBatchSize);
lossVec = zeros(1,numBatches*numMiniBatches);

% set last mini batch size to miniBatchSize if batchSize is a multiple of
% miniBatchSize
if (~lastMiniBatchSize)
    lastMiniBatchSize = miniBatchSize;
end

for i = 1:numBatches
    [data,classes] = read_cifar_data(i);
    data = data/255;    % divide pixel values by 255
    
    for kk = 1:numMiniBatches
        
        % Permute the sample indices in each mini batch.  if/else handles
        % case of last mini batch which could be smaller than the rest.
        if (kk ~= numMiniBatches)
            sampleOrder = randperm(miniBatchSize) + (kk-1)*miniBatchSize;
            mbSize = miniBatchSize;
        else
            sampleOrder = randperm(lastMiniBatchSize) + (kk-1)*miniBatchSize;
            mbSize = lastMiniBatchSize;
        end
        
        % Initialize loss and errors for this mini batch
         mbLoss = 0;
         mbError1 = zeros(numClasses,1);
         mbError2 = zeros(imageSize/poolDim,imageSize/poolDim,channelsOut);
         mbError3 = zeros(imageSize,imageSize,channelsOut);
        tic; 
        for j = 1:mbSize
                       
            % Initialize input layer
            img = reshape(data(sampleOrder(j),:),[imageSize,imageSize,channelsIn]);
            img = transpose_layer(img,channelsIn);
            img = random_flip(img);
            img = random_brightness(img);
            for ii = 1:channelsIn
                singleImg = img(:,:,ii);
                singleImg = singleImg(:);
                adjStdDev = max(std(singleImg),1/imageSize);
                singleImg = (singleImg - mean(singleImg))/adjStdDev;
                img(:,:,ii) = reshape(singleImg,[imageSize,imageSize]);
            end
            
            % create layer struct
            layerIn = struct('height',imageSize,'width',imageSize,'depth',channelsIn,'value',img,'derivative',[]);
            layerIn.value = stochastic_quantize(layerIn.value);
            
            % create label vector
            label = zeros(numClasses,1);
            label(classes(sampleOrder(j))+1) = 1;
            
            conv1 = conv_layer(layerIn,weights1,bias1);
            pool1 = max_pool(conv1,poolDim,poolStride);
            fc1 = fc_layer(pool1,weights2,bias2);
            
            % update counter indicating 1 sample has been processed through FF
            info.iter = info.iter + 1;
            
            % accumulate loss for this mini batch
            mbLoss = mbLoss + report_l2_loss(label,fc1.value);          
            
            % backpropagate errors
            error1 = stochastic_quantize((fc1.value - label).*(fc1.derivative));
            error2 = fc_layer_propagate(mbError1,pool1,weights2);
            error3 = conv_layer_propagate(error2,conv1,poolDim);
            
            % accumulate errors for this mini batch
            mbError1 = stochastic_quantize(mbError1 + error1);
            mbError2 = stochastic_quantize(mbError2 + error2);
            mbError3 = stochastic_quantize(mbError3 + error3);
        end
        timerVal = toc;
        
        % average the loss and errors over this mini batch
        mbLoss = mbLoss/mbSize;
        mbError1 = stochastic_quantize(mbError1/mbSize);
        mbError2 = stochastic_quantize(mbError2/mbSize);
        mbError3 = stochastic_quantize(mbError3/mbSize);
        
        % report loss (this is the average loss over this mini batch)
        info.loss = mbLoss;
        lossVec((i-1)*numMiniBatches+kk) = info.loss;
        fprintf('\n Batch: %d Mini-Batch: %d loss = %.5f (%.4f samples/sec) ',i,kk,info.loss,mbSize/timerVal);      
        
        % update weights and biases (done once per mini batch);
        [weights2,bias2] = fc_layer_update(weights2,bias2,mbError1,pool1,learningRate);
        [weights1,bias1] = conv_layer_update(weights1,bias1,mbError3,layerIn,learningRate);
        
    end
    
end

% Write out csv of lossVec
csvwrite(matlabloss_0p003.csv,lossVec);

% Plot loss of every mini batch
plot(lossVec);
xlabel('iterations (x128)');
ylabel('loss')
title(['2-norm Loss, Mini-Batch Size=' num2str(miniBatchSize) ', Learning Rate=' num2str(learningRate)]);