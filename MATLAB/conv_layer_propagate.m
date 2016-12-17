function [ errorOut ] = conv_layer_propagate( errorIn , layer , poolDim )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

for i = 1:layer.depth
    errorOut(:,:,i) = layer.derivative(:,:,i).*kron(errorIn(:,:,i),ones(poolDim));
end

end

