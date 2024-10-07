
%%%%%% Project assignement 1 %%%%%%
% By Lucie Della Negra
% lude5925@student.su.se
% Due : September 22nd
% Last modification : September 10th

%% Get Started

clear;
load mnistdata;
 
% Visualize a selected train/test digit

figure(1)
n = 6;
for i = 1:n*n

   digit = train8(i,:);
   %digit = test8(i,:);

   digitImage = reshape(digit,28,28);

   subplot(n,n,i);
   image(rot90(flipud(digitImage),-1));
   colormap(gray(256));
   axis square tight off;

end

% Visualize the average train digits

T(1,:) = mean(train0);
T(2,:) = mean(train1);
T(3,:) = mean(train2);
T(4,:) = mean(train3);
T(5,:) = mean(train4);
T(6,:) = mean(train5);
T(7,:) = mean(train6);
T(8,:) = mean(train7);
T(9,:) = mean(train8);
T(10,:) = mean(train9);

for i = 1:10
    digitImage_mean(:,:,i) = reshape(T(i,:),28,28);
end

figure(2)
for i = 1:10
    subplot(2,5,i)
    image(rot90(flipud(digitImage_mean(:,:,i)),-1));
    colormap(gray(256));
    axis square tight off;
end

whos('-file','mnistdata.mat')

%%%%%%%%%% Part 1 : The centroid method
%% Step a : Calculate the distance between the means and the test image

z = double(test7(55,:));
dist = zeros(10,1);
for k=1:10
    dist(k) = norm( z - T(k,:) );
end
dist


%% Step b: Classify the test set of digit

function labels = compute_labels_centroid(digits, average_digits)
    digits = double(digits);
    n = length(digits);% Number of sample
    labels = zeros(n,1);% Output

    % For each digit
    for  i = 1:n
        % Compute the 2-norm distance between this digit and the 10 average
        % train digits
        dist = zeros(10,1);

        for k=1:10
            dist(k) = norm(digits(i,:)-average_digits(k,:));
        end

        % Keep the index of the minimum distance 
        [~,I] = min(dist);
        labels(i) = I-1; % Minus 1 because indexes start at 1 in Matlab
    end
    
end 

% Test it !
compute_labels_centroid(test1, T)


%% Step c: Report the success rate (correct/total) of each digit

function report_success_rate(computational_method, method_param)
    accuracy = zeros(10,1);% Results matrix
    results = "";% Text output for printing results

    % For each test set :
    for test_set_index = 0:9
        test_set = evalin('base', "test"+test_set_index);% Get the test set
        labels = computational_method(test_set, method_param);% Compute labels
        accuracy(test_set_index+1) = sum(labels==test_set_index)/length(labels);% Compute accuracy
        results = results + "Accuracy for digit = "+ test_set_index + " : " + accuracy(test_set_index+1) + newline;% Prepare print
    end

    results
end

% Accuracy results for the centroid method :
report_success_rate(@compute_labels_centroid, T)

% Despite some relatively low accuracy for digit = 5, 8 or 2, the centroid
% method seems to be efficient on this set of data with accuracy rate
% globally superior to 80%

%%%%%%%%%% Part 2: The PCA Method
%% Step a: Find the principal components of the training set
% Given functions
[U3,~,~] = svds(double(train3'), 5) ;
size(U3)

%viewdigit( U3(:,1) );
%viewdigit( U3(:,2) ) ;

basis_len = 5;
Us=zeros( 28*28, basis_len, 10);

for k=1:10
    % go through each digit 0 to 9
    s = strcat('train',num2str(k-1));
    A = double(eval(s));
    % and get first 5 singular vector
    [U,~,~] = svds( A', basis_len );
    Us(:,:,k)=U;
end

z = double(test4(14,:))';
dist = zeros(10,1);
for k=1:10
    Uk = Us(:,:,k);
    dist(k) = norm( z - Uk*(Uk'*z) );
end
dist

%% Function to code

function labels = compute_labels_by_PCA(digits, digits_components)
    digits = double(digits)';
    n = length(digits);% Number of sample
    labels = zeros(n,1);% Output

    % For each digit
    for  i = 1:n
        % Compute dist between digits components and current digit
        dist = zeros(10,1);

        for k=1:10
            Uk = digits_components(:,:,k);% Get the digits components for the current digit
            dist(k) = norm( digits(:, i) - Uk*(Uk'*digits(:, i)) );% Compute the 2-norm distance
        end

        % Keep the index of the minimum distance 
        [~,I] = min(dist);
        labels(i) = I-1; % Minus 1 because indexes start at 1 in Matlab
    end
end 

% Test it !
compute_labels_by_PCA(test0, Us)


%% Step b: Test and report the success rate

report_success_rate(@compute_labels_by_PCA, Us)

% According to the accuracy rates of PCA, this method seems to be more
% efficient than the previous one (centroid method). With rates no lower
% than 88%, we can say that PCA is adapted for the analyse of this dataset.
% We can however, try to improve these rates by changing the length of the
% basis : 


%% See eigenvalues of singular values for basis_len = 20

basis_len = 20;
Us = zeros(28*28, basis_len, 10);
eigenvalues = zeros(basis_len, 10);% Eigenvlaues for each digits

% Same PCA calculation for each digit
for k=1:10
    s = strcat('train',num2str(k-1));
    A = double(eval(s));
    [U,S,~] = svds( A', basis_len);
    eigenvalues(:,k)=diag(S); % Record the values of eigenvalues
    Us(:,:,k)=U;
end

% Plot results on the same graph for every digit
figure(3)
hold on
for i=1:size(eigenvalues, 2)
    plot(eigenvalues(:,i))
end

%% Using the mean for finding basis_len

figure(4)
plot(mean(eigenvalues, 2))

% basis_len = 5 seems to be actually a pretty good choice as the "highest" visible
% elbows are at basis_len = 4 and 6.

%% Looking at the detail for each digit

%plot(eigenvalues(:,1)) % 5 seems to be ok

%plot(eigenvalues(:,2)) % 8

%plot(eigenvalues(:,3)) % 12

%plot(eigenvalues(:,4)) % 7

%plot(eigenvalues(:,5)) % 13

%plot(eigenvalues(:,6)) % 10

%plot(eigenvalues(:,7)) % 8

%plot(eigenvalues(:,8)) % 10

%plot(eigenvalues(:,9)) % 8

%plot(eigenvalues(:,10)) % 9

basis_lengths = [5 8 12 7 13 10 8 10 8 9];
% These values for basis_len are quite arbitrary chosen as the elbows are
% often not "well marked"


%% Compute accuracy with adaptive basis_len parameter

% Re-compute the SVDS
Us = zeros(28*28, max(basis_lengths), 10);

for k=1:10
    s = strcat('train',num2str(k-1));
    A = double(eval(s));
    [U,~,~] = svds( A', basis_lengths(k));
    U(28*28, max(basis_lengths)) = 0; % Expanding matrix U so it has the same size as Us
    Us(:,:,k)=U;
end

report_success_rate(@compute_labels_by_PCA, Us)

% Determination of basis_len by plotting the eigenvalues allows to obtain
% better results or at least equivalent results for some digits. However,
% for some others, the results are affected by the arbitrary choose of
% basis_len which is difficult to justify. Moreover, larger basis lead to
% higher computation time.