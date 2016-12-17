function [ weightsNew , biasNew ] = conv_layer_update( weightsOld , biasOld , error , layerIn , learningRate )

% Initialize new weights and biases as their old counterparts
weightsNew = weightsOld;
biasNew = biasOld;

% Update weights
for j = 1:weightsNew.out
    for i = 1:weightsNew.in
        deltaW = stochastic_quantize(learningRate*stochastic_quantize(rot90(conv2(layerIn.value(:,:,i),rot90(error(:,:,j),2),'valid'),2)));
        weightsNew.value(:,:,i,j) = stochastic_quantize(weightsOld.value(:,:,i,j) - deltaW);
    end
end

% Update biases
for i = 1:biasNew.dim
    biasNew.value(i) = stochastic_quantize(biasOld.value(i) - stochastic_quantize(learningRate*sum(sum(error(:,:,i)))));
end

end

