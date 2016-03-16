function p=predict(theta,X)
%Rolling theta
theta1=reshape(theta(1:20),4,5);
theta2=reshape(theta(21:35),3,5);

X=[ones(size(X,1),1) X];
disp(size(X));
disp(size(theta1'));

p1=sigmoid(X*theta1');
p1a= [ones(size(p1,1),1) p1];
p2=sigmoid(p1a*theta2');

[pEl,p]=max(p2,[],2); %gives the maximum for each row





end
