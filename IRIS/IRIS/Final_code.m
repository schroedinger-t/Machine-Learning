%load data
[loadTable,X,y]=loadData;

%Randomise the whole data set
[Xrand, yrand]=randomize(X,y);

%calculate the cost function
theta1=zeros(4,5);
theta2=zeros(3,5);
initial_theta= [theta1(:);theta2(:)];
lambda=2;
fprintf('\nTraning the Neural network...\n');



%optimise function
options= optimset('GradObj','on','MaxIter',75);
[Theta,FinalGrad]= fmincg(@(t)(costFunction(t,X,y,lambda)),initial_theta,options);

%predict
p=randperm(50);

Test=X(101:150,:);
Test=Test(p,:);

TestSol=y(101:150,:);
TestSol=TestSol(p,:);

pred=predict(Theta,Test);
disp(pred)
r=size(pred,1);
correct=0;
for i=1:r
    if ((pred(r)==1 && strcmp(TestSol(r,:),'Iris-setosa'))||(pred(r)==2 && strcmp(TestSol(r,:),'Iris-versicolor'))||(pred(r)==1 && strcmp(TestSol(r,:),'Iris-virginica')))
        correct=correct+1;
    end
end
Accuracy=(correct./r).*100;
fprintf('The accuracy of the ANN is %d', Accuracy);