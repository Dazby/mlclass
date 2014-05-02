function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

%Submission
C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

%Unit test
%C_vect = [0.01; 1; 3];
%sigma_vect = [0.01; 1; 3];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%error_master = zeros(9, 3);
%counter = 1;
min_error = 1;
for c = 1:size(C_vect)
	for s = 1:size(sigma_vect)
		model = svmTrain(X, y, C_vect(c), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(s)));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if (error < min_error)
			min_error = error;
			C = C_vect(c);
			sigma = sigma_vect(s);
		end
		%error_master(counter,:) = [C sigma error];
		%counter = counter + 1;
	end
end




% =========================================================================

end
