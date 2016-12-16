function [ layerDerivative ] = relu_derivative( layer )
% Takes as input a layer struct, and for each output map, returns the
% derivative of the ReLU activation in the struct layerDerivative.  This
% amounts to 0 if the pixel's activation is < 0 and 1 otherwise.

layerDerivative = struct('height',layer.height,'width',layer.width,'depth',layer.depth,'value',[]);
layer.value = reshape(layer.value,[layer.height,layer.width,layer.depth]);
layerDerivative.value = zeros(size(layer.value));

for i = 1:layer.depth
    layerDerivative.value(:,:,i) = (layer.value(:,:,i) >= 0); 
end

end

