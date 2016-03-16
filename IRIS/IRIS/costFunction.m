function [J, Grad]=costFunction(initial_theta,X,y,lambda) 
%here y should be passed after string comparison, (one vs all)
%Perform forward propogation then perform backward propogation 
%calculate the error function and increment it using the delta values and
%the error values

%code for forward propogation
%no of hidden layers=1
%for the first hidden layer

m=size(X,1);
%Rolling theta
Xa=[ones(size(X,1),1) X];
theta1=reshape(initial_theta(1:20),4,5);
theta2=reshape(initial_theta(21:35),3,5);
 
%convert y from text to a 150x3 binary matrix
yCalculate=zeros(size(y,1),3);

n=find(strcmp(y,'Iris-setosa'));
for w=1:size(n,1)
    yCalculate(n(w),:)=[1 0 0];
end


n=find(strcmp(y,'Iris-virginica'));
for w=1:size(n,1)
    yCalculate(n(w),:)=[0 0 1];
end

n=find(strcmp(y,'Iris-versicolor'));
for w=1:size(n,1)
    yCalculate(n(w),:)=[0 1 0];
end





a1=Xa*theta1'
z1=sigmoid(a1); %hypothesis for the hidden layer
z1a=[ones(size(z1,1),1) z1]; %add the bias parameter

a2=z1a*theta2';
z2=sigmoid(a2); %hypothesis for the second layer

delta3=z2-yCalculate;

theta2a=theta2;
theta2a(:,1)=0;   %notdoing the calculations for the bias functions.
delta2=((ones(size(z2))-z2)*z2')*(delta3*theta2a);
theta1a=theta1;
theta1a(:,1)=0;

%Calculation for the cost function
y1=ones(size(yCalculate(:,1)));
z21=ones(size(z2(:,1)));
J1=0;
for i=1:3
    for j=1:4
        for k=1:3
            J= J1 - (yCalculate(:,i)'*log(z2(:,i)) + (y1-yCalculate(:,i))'*log(z21-z2(:,1)))/m + lambda*(theta1a(j,:)*theta1a(j,:)')/2/m + lambda*(theta2a(k,:)*theta2a(k,:)')/2/m;
        end
    end
end

    
%Calculation for the gradient for total 3 layers
grad1= (X'*delta2)/m + lambda*theta1a;
grad2= (delta3'*z1a)/m + lambda*theta2a;
Grad=[grad1(:);grad2(:)];






%=======================================================
end