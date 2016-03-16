% clear all
% close all
% clc

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

for i_CV=1:5
    [Xrand, yrand]=randomize(X,yData) %%%%Randomize the dataset

    %%%Training File%%%%%%
    TrainX=Xrand(1:100,:);
    TrainT=yrand(1:100,:);
    TrainingFile= cat(2,TrainX,TrainT);

    %%%%Testing file%%%%%%
    TestX=Xrand(101:150,:);
    TestT=yrand(101:150,:);
    TestFile=cat(2,TestX, TestT);

    %Creating an Array for TrainingAccuracy and TestingAccuracy
    TrainingAccuracy= zeros(10,1);
    TestingAccuracy=zeros(10,1);
    Xaxis=zeros(10,1);
    for i_avg=1:10
        [TrainingTime,TestingTime, TrainingAccuracy(i_avg,1), TestingAccuracy(i_avg,1)]=elm_classification(TrainingFile,TestFile,'sig',20,3);
        Xaxis(i_avg,1)=i_avg;
    end
    TrainingAvg(1,1)=mean(TrainingAccuracy);
    TestingAvg(1,1)=mean(TestingAccuracy);
    ind=ones(10,1);
    TrainingAvg=TrainingAvg(ind,:); %%%%Line for the average on the plot
    TestingAvg=TestingAvg(ind,:);   %%%%Line for the average on the plot
   
    figure
    %figure.Name='Training Data';
   % figure('Name', 'Training Data', 'NumberTitle','off');
    subplot(3,2, i_CV);
    [ax,b,p]=plotyy(Xaxis,TrainingAccuracy,Xaxis,TrainingAvg,'bar', 'plot');
    title(['The Training Accuracy for ', num2str(i_CV), ' Cross validation iteration']);
    xlabel('Number of Iteration');
    ylabel(ax(1),'Training Accuracy');
    ylabel(ax(2),' ');
    
    figure;
    figure.Name='Testing Data';
    %figure('Name','Testing Data','NumberTitle','off');
    subplot(3,2,i_CV);
    [ax,b,p]=plotyy(Xaxis,TestingAccuracy,Xaxis,TestingAvg,'bar', 'plot');
    title(['The Testing Accuracy for ',  num2str(i_CV), ' Cross validation iteration']);
    xlabel('Number of Iteration');
    ylabel(ax(1),'Testing Accuracy');
    ylabel(ax(2),' ');

end


