function [ layerTransposeValue ] = transpose_layer( layerValue , depth )
% Returns a multidimensional array whose maps are the transpose of each feature map
% in layerValue.

layerTransposeValue = layerValue;
for i = 1:depth
    layerTransposeValue(:,:,i) = layerTransposeValue(:,:,i)';
end

end

