function [Xrand,yrand]=randomize(x,y)
m=size(x,1);
p=randperm(m);
Xrand=x(p,:);
yrand=y(p,:);




%==============================================================================
end