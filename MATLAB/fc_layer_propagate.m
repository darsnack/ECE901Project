function [ errorOut ] = fc_layer_propagate( errorIn , layer , weights )
%

% transpose the value fields for ease of vectorization
activationDerivative = transpose_layer(layer.derivative,layer.depth);

% vectorize the derivative maps and backpropagate
errorOutVec = stochastic_quantize((weights.value'*errorIn).*activationDerivative(:));

% cubify the errorOutVec vector
dim = layer.height;
errorOut = reshape(errorOutVec,[dim,dim,layer.depth]);

% transpose errorOut to obtain correct ordering
errorOut = transpose_layer(errorOut,layer.depth);

end

