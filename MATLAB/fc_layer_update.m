function [ weightsNew , biasNew ] = fc_layer_update( weightsOld , biasOld , error , layerIn , learningRate )

% Initialize new weights and biases as their old counterparts
weightsNew = weightsOld;
biasNew = biasOld;

% Transpose the feature maps going into the fc layer for ease of
% vectorization
layerInTranspose = layerIn;
layerInTranspose.value = transpose_layer(layerIn.value,layerIn.depth);

% Update weights
weightsNew.value = weightsOld.value - stochastic_quantize(learningRate*stochastic_quantize(error*layerInTranspose.value(:)'));

% Update biases
biasNew.value = stochastic_quantize(biasOld.value - stochastic_quantize(learningRate*error));

end

