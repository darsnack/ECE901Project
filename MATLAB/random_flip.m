function [ imgOut ] = random_flip( imgIn )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

imgOut = imgIn;
if (rand(1) < 0.5)
    for i = 1:3
        imgOut(:,:,i) = fliplr(imgIn(:,:,i));
    end
end

end

