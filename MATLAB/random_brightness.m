function [ imgOut ] = random_brightness( imgIn )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

imgOut = imgIn;
delta = 0.5*rand(1)-0.25;
if (rand(1) < 0.5)
    for i = 1:3
        imgOut(:,:,i) = imgIn(:,:,i)+delta;
        imgOut(:,:,i) = max(zeros(size(imgOut(:,:,i))),imgOut(:,:,i));
        imgOut(:,:,i) = min(ones(size(imgOut(:,:,i))),imgOut(:,:,i));
    end
end

end

