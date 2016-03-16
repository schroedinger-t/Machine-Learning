clear all
close all
clc

%%%%%Preparing the data to be passed on to the ELM function%%%%%

[Table,X,y]=loadData

%convert y from text to a 150x3 binary matrix
yData=zeros(size(y,1),3);

n=find(strcmp(y,'Iris-setosa'));
for w=1:size(n,1)
    yData(n(w),:)=[1 0 0];
end


n=find(strcmp(y,'Iris-virginica'));
for w=1:size(n,1)
    yData(n(w),:)=[0 0 1];
end

n=find(strcmp(y,'Iris-versicolor'));
for w=1:size(n,1)
    yData(n(w),:)=[0 1 0];
end

[Xrand, yrand]=randomize(X,yData) %%%%Randomize the dataset

TrainX=Xrand(1:100,:);
TrainT=yrand(1:100,:);
TrainingFile= cat(2,TrainX,TrainT);

% save('file_train.mat','TrainingFile');

TestX=Xrand(101:150,:);
TestT=yrand(101:150,:);
TestFile=cat(2,TestX, TestT);

% save('file_test.mat','TestFile');
TrainingAvg=zeros(10,1);
TestingAvg=zeros(10,1);
TrainingAccuracy=zeros(10,1);
TestingAccuracy=zeros(10,1);
Xaxis=zeros(10,1);

for i_nH=1:15
    for jAvg=1:10
        [TrainingTime,TestingTime, TrainingAccuracy(jAvg,1), TestingAccuracy(jAvg,1)]=elm_classification(TrainingFile,TestFile,'sig',(3*i_nH),3);
    end
    TrainingAvg(i_nH,1)=mean(TrainingAccuracy);
    TestingAvg(i_nH,1)= mean(TestingAccuracy);
    Xaxis(i_nH,1)= 3*i_nH;
end

figure;
plot(Xaxis,TrainingAvg,'--ro',Xaxis, TestingAvg,':b*');
title('Optimal Number of Hidden Neurons');
xlabel('Number of hidden Neurons');
ylabel('% Accuracy');
ylim([85,100]);

% fprintf('\n\nTraining Time is %E', TrainingTime);
% fprintf('\n\nTesting Time is %E', TestingTime);

%%%%%precision%%%%%

