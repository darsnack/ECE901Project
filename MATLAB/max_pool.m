function [ layerOut ] = max_pool( layerIn, dim, stride )
%Takes as input the struct layerIn, containing a
%height-by-width-by-depth-dimensional array in layerIn.value.  Performs
%max-pooling operation, returning output struct layerOut containing
%height/dim-by-width/dim-depth array.  Stride length of stride used.

layerOut.height = layerIn.height/dim;
layerOut.width = layerIn.width/dim;
layerOut.depth = layerIn.depth;
layerOut.value = zeros(layerOut.height,layerOut.width,layerOut.depth);

for i = 1:layerOut.depth
    for j = 1:layerOut.width
        for k = 1:layerOut.height
            % pool is the dim-by-dim submatrix.  We pick out the max of
            % these values.
            pool = layerIn.value(k*dim-stride:(k+1)*dim-stride-1,...
                j*dim-stride:(j+1)*dim-stride-1,i);
            layerOut.value(k,j,i) = max(pool(:));
        end
    end
end


end

