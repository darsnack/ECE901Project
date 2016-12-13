function [ activation ] = relu_activate( preActivation )
% preActivation is a feature map.  This function activates each pixel of
% the image using the ReLu activation function.

[dim1,dim2] = size(preActivation);
activation = zeros(dim1,dim2);

for i = 1:dim1
    for j = 1:dim2
        pixel = preActivation(i,j);
        if (pixel >= 0)
            activation(i,j) = preActivation(i,j);
        end
    end
end

end

