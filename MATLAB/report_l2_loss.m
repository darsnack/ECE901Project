function [ loss ] = report_l2_loss( label, prediction )
% Given the true image label in a 1-hot vector 'labels', and output prediction, computes l2 loss between the two.

if (length(label) ~= length(prediction))
   error('Error: argument "labels" must be 1-hot vector of same length as "prediction".') 
end

loss = 0.5*norm(label-prediction)^2;

end

