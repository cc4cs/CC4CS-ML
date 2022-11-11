
inputResume=readtable('inputResume.csv','Delimiter',',');

inputName = inputResume(:,1);

scalarColumn = 3;  % Scalar Variable
sizeColumn = 2;   % Array/Matrix Size

folderDevRes = strcat('inputResumeScalarCSV');
[status, msg, msgID] = mkdir(folderDevRes)

% SCALAR
%{
inputScalarValue = inputResume(:,scalarColumn);
inputScalarValue = [inputName inputScalarValue];

imageFileName = strcat(folderDevRes,'/ScalarVariable.csv');
imageFileName = char(imageFileName);
writetable(inputScalarValue,imageFileName,'WriteVariableNames',0);
%}

% VECTOR OR MATRIX SIZE

inputScalarValue = inputResume(:,sizeColumn);
inputScalarValue = [inputName inputScalarValue];

imageFileName = strcat(folderDevRes,'/ScalarSize.csv');
imageFileName = char(imageFileName);
writetable(inputScalarValue,imageFileName,'WriteVariableNames',0);