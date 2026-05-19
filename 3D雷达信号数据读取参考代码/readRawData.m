function [retVal,para] = readRawData(filename,para)
%READRAWDATA read raw data from DCA1000
%   input: filename,para
%   output:retVal
if ~isfield(para,"FrameStart")
    para.FrameStart=1;
end
if ~isfield(para,"FrameLength")
    para.FrameLength=para.FrameNum-para.FrameStart+1;
end

FrameNum=para.ADCSamples*para.ChirpNum*para.GroupNum*para.numRX;
fid = fopen(filename,'r');
fseek(fid,(2*FrameNum*(para.FrameStart-1))*2,'bof');
adcData = fread(fid,[4,FrameNum*(para.FrameLength)/2],'int16');
fclose(fid);
retVal = zeros(2,FrameNum*(para.FrameLength)/2);
retVal(1,:) = adcData(1,:)+1j*adcData(3,:);
retVal(2,:) = adcData(2,:)+1j*adcData(4,:);
retVal=reshape(retVal,para.ADCSamples,para.numRX*para.GroupNum,para.ChirpNum,para.FrameLength);

end

