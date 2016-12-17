function [ layerOut ] = fc_layer( layerIn , weights , bias )
% Takes layerIn struct as input and produces layerOut.  layerOut is fully
% connected with layerIn neurons.

layerOut.height = 1;
layerOut.width = 1;
layerOut.depth = weights.out;
layerOut.value = zeros(layerOut.height,layerOut.width,layerOut.depth);

layerInTranspose = layerIn;
layerInTranspose.value = transpose_layer(layerIn.value,layerIn.depth);

preActivation = weights.value*layerInTranspose.value(:) + bias.value;

layerOut.derivative = relu_derivative(preActivation);
layerOut.value = relu_activate(preActivation);  


end

