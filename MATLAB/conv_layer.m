function [ layerOut ] = conv_layer( layerIn , weights , bias )
%Takes as input the struct layerIn, containing a
%height-by-width-by-depth-dimensional array in layerIn.value.  Performs
%convolution of kernel with each channel in layerIn, and after adding bias,
%activates the output.  Returns output struct layerOut containing
%height-by-width-#channels array.


layerOut.height = layerIn.height;
layerOut.width = layerIn.width;
layerOut.depth = weights.out;
layerOut.value = zeros(layerOut.height,layerOut.width,layerOut.depth);
layerOut.derivative = zeros(layerOut.height,layerOut.width,layerOut.depth);

if (weights.in ~= layerIn.depth)
   error('Error: Number of slices in input layer must be == number of slices in kernel.') 
end

for i = 1:layerOut.depth
    for j = 1:layerIn.depth
        layerOut.value(:,:,i) = layerOut.value(:,:,i) + conv2(layerIn.value(:,:,j),weights.value(:,:,j,i),'same');        
    end
    preActivation = layerOut.value(:,:,i) + bias.value(i);
    layerOut.derivative(:,:,i) = relu_derivative(preActivation);
    layerOut.value(:,:,i) = relu_activate(preActivation);
end

end
