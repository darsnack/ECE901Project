function [ errorOut ] = fc_layer_propagate( errorIn , layer , weights )
%

% layerDerivative returns the derivative of layer
layerDerivative = relu_derivative(layer);

% transpose the value fields for ease of vectorization
layerDerivativeTranspose = layerDerivative;
layerDerivativeTranspose.value = transpose_layer(layerDerivative.value,layer.depth);

% vectorize the derivative maps and backpropagate
errorOutVec = (weights.value'*errorIn).*layerDerivative(layer).*layerDerivativeTranspose.value(:);

% cubify the errorOutVec vector
dim = layer.height;
errorOut = reshape(errorOutVec,[dim,dim,layer.depth]);

% transpose errorOut to obtain correct ordering
errorOut = transpose_layer(errorOut,layer.depth);

end

