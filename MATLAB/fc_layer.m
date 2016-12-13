function [ layerOut ] = fc_layer( layerIn , weights , bias )
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here

layerOut.height = 1;
layerOut.width = 1;
layerOut.depth = weights.channelsOut;
layerOut.value = zeros(layerOut.height,layerOut.width,layerOut.depth);

for i = 1:layerOut.depth
    for j = 1:layerIn.depth
        x = layerIn.value(:,:,j);
        w = weights.value(:,:,j);
        layerOut.value(:,:,i) = layerOut.value(:,:,i) + x(:)'*w(:);
    end
    preActivation = layerOut.value(:,:,i) + bias(i);
    layerOut.value(:,:,i) = relu_activate(preActivation);    
end

end

