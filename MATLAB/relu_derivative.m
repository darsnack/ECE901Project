function [ layerDerivativeValue ] = relu_derivative( layerValue )
% Takes as input an array , and returns the
% derivative of the ReLU activation in the array layerDerivativeValue.  This
% amounts to 0 if the pixel's activation is < 0 and 1 otherwise.

layerDerivativeValue = (layerValue >= 0); 

end

