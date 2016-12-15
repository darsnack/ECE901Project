function [ weightsNew , biasNew ] = fc_layer_update( weightsOld , biasOld , error , layerIn , learningRate )
%

layerInTranspose = layerIn;
layerInTranspose.value = transpose_layer(layerIn.value,layerIn.depth);

weightsNew = weightsOld - learningRate*(error*layerInTranspose(:)');

biasNew = biasOld - learningRate*error;

end

