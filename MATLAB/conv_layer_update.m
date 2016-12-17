function [ weightsNew , biasNew ] = conv_layer_update( weightsOld , biasOld , error , layerIn , learningRate )

% Initialize new weights and biases as their old counterparts
weightsNew = weightsOld;
biasNew = biasOld;

% Update weights
for j = 1:weightsNew.out
    for i = 1:weightsNew.in
        deltaW = learningRate*rot90(conv2(layerIn.value(:,:,i),rot90(error(:,:,j),2),'valid'),2);
        weightsNew.value(:,:,i,j) = weightsOld.value(:,:,i,j) - deltaW;
    end
end

% Update biases
for i = 1:biasNew.dim
    biasNew.value(i) = biasOld.value(i) - learningRate*sum(sum(error(:,:,i)));
end

end

