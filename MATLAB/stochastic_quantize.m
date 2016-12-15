% The following function will quantize real numbers (doubles)
% using stochastic rounding.
% The fixed point format is <int_length, frac_length> as specified
% by the parameters below. One bit is assumed to be a sign bit.
% Any values above and below the range of the fixed point number format
% are saturated to the maximum and minimum values.
% Works for any matrix, vector, or scalar.
function y = stochastic_quantize(x)

% Specify the word length and fractional length as desired
word_length = 16;
frac_length = 14;
int_length = word_length - frac_length - 1;
epsilon = 2^(-frac_length);
min_val = -(2^int_length);
max_val = 2^int_length - epsilon;

% Initialize y to the same size as x
y = zeros(size(x));

% Saturate values below min_val
idx = find(x <= min_val);
y(idx) = min_val;

% Saturate values above max_val
idx = find(x >= max_val);
y(idx) = max_val;

% Find values to quantize
idx = find((x > min_val) & (x < max_val));
% Push all the fractional bits into the integer domain
% Everything below the decimal point is now excess
y(idx) = x(idx) .* (2^frac_length);
% Round according the distance of the number from the floor value
floor_val = floor(y(idx));
ceil_val = ceil(y(idx));
p = y(idx) - floor_val;
r = rand(length(idx), 1);
ceil_idx = find(r < p);
floor_idx = find(r >= p);
y(idx(floor_idx)) = floor_val(floor_idx);
y(idx(ceil_idx)) = ceil_val(ceil_idx);
% Push fractional bits back into the fractional domain
y(idx) = y(idx) .* epsilon;

end