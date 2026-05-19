function para = readPara(fpath)
%READPARA read parameter from DCA1000 LogFile
%   input  : filename
%   output : para
%   example: para=readPara("1_LogFile.txt")
fidin = fopen(fpath,'r');
n=0;
while ~feof(fidin)  % �ж��Ƿ�Ϊ�ĵ�ĩβ��
tline=fgetl(fidin); % ��һ�У�
n=n+1;
if contains(tline,'ProfileConfig')% Ѱ�ң�ProfileConfigλ�á�
    n1 = n;
elseif contains(tline,'AdvancedFrameConfig')% Ѱ�ң�FrameConfigλ�á�
    n2 = n;
end
% if contains(tline,'ChirpConfig')% Ѱ�ң�FrameConfigλ�á�
%         
% end

end
fclose(fidin);
n=0;
fidin = fopen(fpath,'r');
while ~feof(fidin)  % �ж��Ƿ�Ϊ�ĵ�ĩβ��
tline=fgetl(fidin); % ��һ�У�
n=n+1;

    if n == n1% Profile λ�ã�
        a=strfind(tline,',');% ��λ���ţ�
        para.IDLEtime = str2double(tline(a(3)+1:a(4)-1))*1e-8; % ����ʱ��
        para.STARTtime = str2double(tline(a(4)+1:a(5)-1))*1e-8;% ADC ��ʼ����ʱ��
        para.ENDtime = str2double(tline(a(5)+1:a(6)-1))*1e-8;  % ����ֹͣʱ��
        % 60G->36210  77G->48279
        para.FrequencySlope = str2double(tline(a(8)+1:a(9)-1))*36210/1e6*1E12;% ��Ƶб��
        para.ADCSamples = str2double(tline(a(10)+1:a(11)-1));% ADC ������
        para.Fs = str2double(tline(a(11)+1:a(12)-1))*1e3;% ����Ƶ��
        para.Chirptime = para.IDLEtime+para.ENDtime;% Chirp ʱ��
    end
    
    
    if n==n2% advancedFrame λ��
        a=strfind(tline,',');% ��λ���ţ�
        para.FrameNum = str2double(tline(a(39)+1:a(40)-1));% ֡����
        para.GroupNum = str2double(tline(a(5)+1:a(6)-1));% һ����֡���ٸ� chirp ��
        para.ChirpNum = str2double(tline(a(6)+1:a(7)-1));% һ֡���ٸ���֡
        para.Frameinter = 0.5*str2double(tline(a(7)+1:a(8)-1))*1e-8;% ֡��
        
    end
    
end
fclose(fidin);
para.numRX=4;%����������
%% �����ⲿ�����ڵ�����60G������Ҫ�Լ����£��������ʱ����
para.f0=60e9;%��Ƶ
para.lambda=3e8./para.f0;%����
para.d=para.lambda/2;%�벨��
para.BandWidth=para.FrequencySlope*para.ADCSamples/para.Fs;%��Ч����
para.dr=3e8/para.BandWidth/2;%����ֱ���?
para.df=1/para.Chirptime/para.GroupNum;%�����ղ���Ƶ��
para.dv=para.lambda*para.df/2/para.ChirpNum;%RD�׵��ٶȷֱ���
end

