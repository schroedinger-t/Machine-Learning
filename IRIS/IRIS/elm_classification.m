function [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy]=elm_classification(TrainingData, TestingData,Func,NumberofHiddenNeurons, NumberofLabels)
% TrainingData=load(TrainingFile,'-mat');
% TestingData=load(TestingFile,'-mat');

n=size(TrainingData,2);

disp(n);

NumberofInputs=(n-NumberofLabels);

disp(NumberofInputs);

P=TrainingData(:,1:NumberofInputs);
T=TrainingData(:,(NumberofInputs+1):n)*2-1; %data and target are processed before passing into the function
Test_P=TestingData(:,1:NumberofInputs);
Test_T=TestingData(:,(NumberofInputs+1):n)*2-1;

%%%%%%%%%%%%%%Initialise input weights and biases%%%%%%%%


startofTrainingTime=cputime;

%the size of input weights is given by the following- no of rows= no of
%inputs and no of columns= no of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputs)*2-1 ;
tempH=P*InputWeight';

BiasofHiddenNeurons=rand(1, NumberofHiddenNeurons);
ind= ones(size(TrainingData,1),1);
BiasMat= BiasofHiddenNeurons(ind, :);

disp(BiasMat)
disp(size(TrainingData,1));
disp(size(BiasMat))
disp(size(tempH));
disp(size(InputWeight));
disp(size(P));

tempH=tempH+BiasMat; %ADDING BIAS FOR ALL HIDDEN NEURONS IN ALL SAMPLES

switch lower(Func)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
end
% disp(H);                     %for debugging
% disp(size(pinv(H)));
% disp(size(T));

%%%%%%%%FINDING OUTPUT WEIGHTS%%%%%%%%%%%%%
OutputWeight=pinv(H)*T;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
endofTrainingTime=cputime;
TrainingTime=endofTrainingTime-startofTrainingTime; 

%%%%TRAINING ACCURACY%%%%%%%%%
Y= H*OutputWeight;

[p1, Pred]=max(Y,[],2);
[tTest, Trainmax]=max(T,[],2);
tpos1=0;
tpos2=0;
tpos3=0;
fpos1=0;
fpos2=0;
fpos3=0;

r=size(Pred,1);
correct_train=0;
for i=1:r
    if (Pred(i)==Trainmax(i))
        correct_train=correct_train+1;
         if Pred(i)==1
            tpos1=tpos1+1;
        end
        if Pred(i)==2
            tpos2=tpos2+1;
        end
        if Pred(i)==3
            tpos3=tpos3+1;
        end
    elseif Pred(i)==1 %%%%%%%%%%%%%%%%%%%%%%%%for False Pos
        fpos1=fpos1+1;
    elseif Pred(i)==2
        fpos2=fpos2+1;
    elseif Pred(i)==3
        fpos3=fpos3+1;
    end
end
TrainingAccuracy=(correct_train./r).*100;
fprintf('\n \n The Training Accuracy is %f',TrainingAccuracy);

TrainingPrecision= ((tpos1./(tpos1+fpos1))+(tpos2./(tpos2+fpos2))+tpos3./(tpos3+fpos3))/NumberofLabels;
fprintf('\n \n The Training Macro Precision is %f',TrainingPrecision);

%%%%%%%%%%%%FOR TESTING%%%%%%%%%%
startTestingTime= cputime;
tempH_Test= Test_P*InputWeight';
numberofTestingData= size(Test_P,1);
ind=ones(numberofTestingData,1);
BiasMat_Test=BiasofHiddenNeurons(ind,:);
tempH_Test=tempH_Test+BiasMat_Test;

switch lower(Func)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        Test_H = 1 ./ (1 + exp(-tempH_Test));
    case {'sin','sine'}
        %%%%%%%% Sine
        Test_H = sin(tempH_Test);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        Test_H = double(hardlim(tempH_Test));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        Test_H = tribas(tempH_Test);
    case {'radbas'}
        %%%%%%%% Radial basis function
        Test_H = radbas(tempH_Test);
end

%%%%%%TESTING ACCURACY%%%%%%
Test_Y=Test_H*OutputWeight;
endTestingTime=cputime;
TestingTime=endTestingTime-startTestingTime;


[pTest, Pred_Test]=max(Test_Y,[],2);
[tTest, Testmax]=max(Test_T,[],2);

Test_r=size(Pred_Test,1);
correct=0;
Test_tpos1=0;
Test_tpos2=0;
Test_tpos3=0;
Test_fpos1=0;
Test_fpos2=0;
Test_fpos3=0;

for i=1:Test_r
    if (Pred_Test(i)==Testmax(i))
        correct=correct+1;
        if (Pred_Test(i)==1)
            Test_tpos1=Test_tpos1+1;
        end
        if (Pred_Test(i)==2)
            Test_tpos2=Test_tpos2+1;
        end
        if (Pred_Test(i)==3)
            Test_tpos3=Test_tpos3+1;
        end
    elseif (Pred_Test(i)==1)
        Test_fpos1=Test_fpos1+1;
    elseif (Pred_Test(i)==2)
        Test_fpos2=Test_fpos2+1;
    elseif (Pred_Test(i)==3)
        Test_fpos3=Test_fpos3+1;
    end
end

TestingPrecision= ((Test_tpos1./(Test_tpos1+Test_fpos1))+(Test_tpos2./(Test_tpos2+Test_fpos2))+(Test_tpos3./(Test_tpos3+Test_fpos3)))./NumberofLabels;
fprintf('\n\n The Testing Macro Precision is %f \n \n',TestingPrecision)

TestingAccuracy=(correct./Test_r).*100;
fprintf('\n \nThe testing accuracy of the elm is %f', TestingAccuracy);

end