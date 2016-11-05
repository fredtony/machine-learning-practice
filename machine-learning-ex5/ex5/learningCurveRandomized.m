function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals. Use average of
%   randomized samples.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

%Randomize values for each i and compute average over 50 iterations
error_terms_train = zeros(m,1);
error_terms_val = zeros(m,1);
for i = 1:m
    for j = 1:100
        indices = randsample(m,i);
        theta = trainLinearReg(X(indices,:),y(indices,1),lambda);
        error_terms_train(i) = linearRegCostFunction(X(indices,:),y(indices,1),theta,0);
        error_terms_val(i) = linearRegCostFunction(Xval,yval,theta,0);
    end;
    error_train(i) = sum(error_terms_train)/i;
    error_val(i) = sum(error_terms_val)/i;
end;

% -------------------------------------------------------------

% =========================================================================

end
