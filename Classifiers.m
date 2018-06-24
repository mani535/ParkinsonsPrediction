%Clearing memory and command window
clear,clc,close all
WB = waitbar(0,'Reading data files');
Data = importdata('data/data.txt');
Out = importdata('data/out.txt');

%% Normalise data
Data = (Data - min(min(Data)))/(max(max(Data)) - min(min(Data))); %Scaled to [0,1]
Data = (Data*2) - 1; %Scaled to [-1,1]
Center_of_mass = mean(mean(Data))

%% Dividing data into training set and testing set
train_ratio = 0.7;
[trainInd,~,testInd] = dividerand(length(Out),train_ratio,0,1-train_ratio);
X = Data(trainInd,:); Y = Out(trainInd,:);
Xtest = Data(testInd,:); Exact = Out(testInd,:);

%% 1) K Nearest Neighbor
waitbar(1/7,WB,'Testing classifier: KNN');

for i =1:20 % Variating fitting parameter to get the best
K_model = fitcknn(X,Y,'NumNeighbors',i);
cross_K = crossval(K_model); %Preform cross validation by 10 folds
score(i,:) = kfoldLoss(cross_K,'mode','individual');  %Estimate loss score from cross validation
end

scoreAV = mean(score,2); %average error from all fold
[~,idx] = min(scoreAV); % Determine best case
tic; %start calculate time taken by classifier
K_model = fitcknn(X,Y,'NumNeighbors',idx);
K_time_train = toc;  %Elapsed time in seconds
W = K_model.W; %Coeficient vector

tic; %start calculate time taken by classifier
K_result = predict(K_model,Xtest); % testing the model
K_time_test = toc;  %Elapsed time in seconds

K_error = ~strcmp(Exact,K_result); %Comparing result to exact result
K_error = sum(K_error); %Counting numbers of mistakes
K_error = 100*K_error/length(Exact); %Represent error as percent

%% 2) Support Vector Machine (SVM)
waitbar(2/7,WB,'Testing classifier: SVM');
tic; %start calculate time taken by classifier
SVM_mdl = fitcsvm(X,Y);
SVM_time_train = toc;  %Elapsed time in seconds
tic; %start calculate time taken by classifier
SVM_res = predict(SVM_mdl,Xtest);
SVM_time_test = toc;  %Elapsed time in seconds
SVM_error = SVM_res-Exact; %Comparing result to exact result
SVM_error = SVM_error~=0; %Vector of mistakes
SVM_error = sum(SVM_error); %Counting numbers of mistakes
SVM_error = 100*SVM_error/length(Exact); %Represent error as percent

%% 3) Random forest
waitbar(3/7,WB,'Testing classifier: Random forest');
% Defining the classifer parameters
opts = struct;        opts.depth = 2; 
opts.numTrees = 10;   opts.numSplits = 2; 
opts.verbose = true;  opts.classifierID = 2;
tic; %start calculate time taken by classifier
tree_mdl = forestTrain(X, Y, opts); %Training the classifier
tree_time_train = toc ; %Elapsed time in seconds
tic; %start calculate time taken by classifier
[tree_res,~] = forestTest(tree_mdl, Xtest, opts);
tree_time_test = toc ; %Elapsed time in seconds
tree_error = tree_res-Exact; %Comparing result to exact result
tree_error = tree_error~=0; %Vector of mistakes
tree_error = sum(tree_error); %Counting numbers of mistakes
tree_error = 100*tree_error/length(Exact); %Represent error as percent

%% 4) Logistic regression
waitbar(4/7,WB,'Testing classifier: Logistic reg.');
tic; %start calculate time taken by classifier
Reg_mdl = fitglm(X,Y); %Training the classifier
Reg_time_train = toc;  %Elapsed time in seconds
tic; %start calculate time taken by classifier
[Reg_res,~] = predict(Reg_mdl,Xtest); % Using model to predict test results
Reg_res=roundn(Reg_res,0); %Rounding results to integers;
Reg_time_test = toc;  %Elapsed time in seconds
Reg_error = Exact-Reg_res; %Comparing result to exact result
Reg_error = Reg_error~=0; %Vector of mistakes
Reg_error = sum(Reg_error); %Counting numbers of mistakes
Reg_error = 100*Reg_error/length(Exact); %Represent error as percent

% %% 5) Neural Network
waitbar(5/7,WB,'Testing classifier: Neural network');
% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
% Train the Network
tic; %start calculate time taken by classifier
[net,tr] = train(net,X',Y');
net_time_train = toc; %Elapsed time_test in seconds
tic; %start calculate time taken by classifier
outputs = round(net(Xtest'));
net_time_test = toc; %Elapsed time_test in seconds
net_error = Exact-outputs';
net_error = net_error~=0; %Vector of mistakes
net_error = 100*abs(sum(net_error))/length(Exact);

%% 6) Decision Tree
waitbar(6/7,WB,'Testing classifier: Tree');
tic; %start calculate time_test taken by classifier
DecTree = fitrtree(X,Y); %Training the classifier
DecTree_time_train = toc; %Elapsed time_test in seconds
tic; %start calculate time_test taken by classifier
DecTree_res = predict(DecTree,Xtest); % Using model to predict test results
DecTree_time_test = toc; %Elapsed time_test in seconds
DecTree_error = Exact-DecTree_res; %Comparing result to exact result
DecTree_error = DecTree_error~=0; %Vector of mistakes
DecTree_error = sum(DecTree_error); %Counting numbers of mistakes
DecTree_error = 100*DecTree_error/length(Exact); %Represent error as percent

%% Showing all results and comparing
waitbar(7/7,WB,'Done');
Ttrain = 1000*[K_time_train,SVM_time_train,tree_time_train,Reg_time_train,net_time_train,DecTree_time_train];
Ttest = 1000*[K_time_test,SVM_time_test,tree_time_test,Reg_time_test,net_time_test,DecTree_time_test];
Er = [K_error,SVM_error,tree_error,Reg_error,net_error,DecTree_error];
Results = [Ttrain;Ttest;(Ttrain+Ttest);Er];
% Create the column and row names in cell arrays 
cnames = {'K Nearest Neighbor','Support Vector Machine','Random Forest',...
          'Logistic Regression','Neural Network','Decision Tree'};
rnames = {'Training time in ms','Testing time in ms','Total time in ms','Error in %'};
f = figure('Position',[20 200 1133 120]);
% Create the uitable
t = uitable(f,'Data',Results,'ColumnWidth',{140},'ColumnName',cnames,'RowName',rnames);
% Set width and height
set(t,'Position',[20 20 1248 96])
close(WB)

%Plotting crossvalidation results for each fold
figure('Position',[26 64 1313 609])
for i=1:10
    subplot(2,5,i),plot(score(:,i)) 
    hold on,plot(score(:,i),'*k'),hold off
    xlabel('Number of neighbors'),ylabel('Error score')
    title(['Fold number' num2str(i)])
end
figure(),plot(scoreAV) %Plotting crossvalidation results
hold on,plot(scoreAV,'*k'),hold off
xlabel('Number of neighbors'),ylabel('Error score')
title('Crossvalidation results average of all folds')



