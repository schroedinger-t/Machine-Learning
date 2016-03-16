function [loadTable,X,y]= loadData

URL=input('URL to take the data from','s');
filename=input('what is the filename for the data to be stored in?', 's');

%urlwrite %webread

file_addr= urlwrite(URL,filename);%import from the website

loadTable= readtable(filename, 'readVariableNames',0); %convert to table

m=size(loadTable,2);
X= table2array(loadTable(:,1:(m-1)));       % convert to matrix
y= table2array(loadTable(:,m));


%==========================================================================
end