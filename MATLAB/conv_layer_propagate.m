function [ errorOut ] = conv_layer_propagate( errorIn , layer , poolDim )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% errorOut = zeros(layer.height,layer.width,layer.depth);
layerDerivative = relu_derivative(layer);

for i = 1:layer.depth
    errorOut(:,:,i) = layerDerivative.value(:,:,i).*kron(errorIn(:,:,i),ones(poolDim));
end

end

