%% batch_extract.m - Extract ALL point clouds
function batch_extract(nPer)
if nargin<1, nPer=100; end
script_dir = fileparts(mfilename('fullpath')); addpath(script_dir);
jsonPath = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonPath);
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');
actions = {'walk','jump','swing','bend','boxing','sit','stand','run','falldown','liedown'};
nR=para.ADCSamples; nL=para.numLoops; nC=para.chirpsPerCycle;
nRX=para.numRXPerDevice; nD=para.numDevices; nTX=para.totalTX; nTRX=para.totalRX;
r_axis=(0:nR-1)'*para.dr; vm=para.lambda/(4*para.Chirptime*nC);
v_axis=((0:nL-1)-nL/2)*(2*vm/nL);
rMin=max(1,round(0.5/para.dr)); rMax=min(nR,round(5/para.dr));
zb=nL/2+1; range_mask=(r_axis>=0.5)&(r_axis<=5.5);
rWin=hanning(nR); dWin=hanning(nL);
tx0=[3,6,9,12]; tx1=[2,5,8,11]; tx2=[1,4,7,10];
devs={'master','slave1','slave2','slave3'};
CFAR_TF=3; SNR_DB=4; ANG_TH=1.5; MAX_D=30; N_A=128;
Z_MIN=0.3; Z_MAX=2.5; X_MAX=1.5; Y_MIN=0.5; Y_MAX=6.0; fs=15; fe=60;
range_mask = (r_axis>=0.5)&(r_axis<=5.5);  %% Wide range for all scenarios
trainDir = fullfile(script_dir, '..', 'train_python', 'pointcloud_data');
if ~exist(trainDir,'dir'), mkdir(trainDir); end
dataFile = fullfile(trainDir, 'all_pointclouds.mat');
if exist(dataFile,'file')
    ld=load(dataFile); all_sc=ld.all_sc; all_lb=ld.all_lb; all_pc=ld.all_pc; nTot=length(all_sc);
else
    all_sc={}; all_lb=[]; all_pc={}; nTot=0;
end
fprintf('===== Batch Extract (up to %d/class) =====\n', nPer);
for a=1:length(actions)
    act=actions{a}; fprintf('\n=== %s ===\n',act);
    for s=1:nPer
        sc=sprintf('CCdata_%s_%04d',act,s);
        if ~exist(fullfile(dataRoot,sc),'dir'), fprintf('  End %s at %d\n',act,s-1); break; end
        fprintf('  [%d] %s...',s,sc); tt=tic;
        adc=cell(nD,1); rdOK=true;
        for d=1:nD
            fp=fullfile(dataRoot,sc,[devs{d} '_0000_data.bin']);
            try; adc{d}=read4DRawData(fp,para,fs:fe); catch; rdOK=false; break; end
        end
        if ~rdOK, fprintf(' READ_ERR\n'); continue; end
        nF=size(adc{1},4); if nF<5, fprintf(' FEW\n'); continue; end
        all_pts=[];
        for fi=1:nF
            rfft=cell(nD,1); for d=1:nD, frm=squeeze(adc{d}(:,:,:,fi)); rfft{d}=fft(frm.*rWin.',[],2); end
            rc=cell(nTX,1);
            for tx=1:nTX, di=para.txChirpMap(tx,1); rt=reshape(rfft{di},nRX,nR,nC,nL); rd=squeeze(rt(:,:,tx,:)); rd=rd-mean(rd,3); rd=fft(rd.*reshape(dWin,1,1,nL),[],3); rc{tx}=fftshift(rd,3); end
            rP=zeros(nR,nL); for k=1:length(tx0), rP=rP+squeeze(mean(abs(rc{tx0(k)}).^2,1))+squeeze(mean(abs(rc{tx1(k)}).^2,1)); end; rP=rP/(2*length(tx0));
            gr=4;gd=2;tr=8;td=4;kr=2*tr+2*gr+1;kd=2*td+2*gd+1; kern=ones(kr,kd);kern(tr+1:tr+2*gr+1,td+1:td+2*gd+1)=0; nA=conv2(rP,kern,'same')/sum(kern(:)); det=rP>(nA*CFAR_TF);e=tr+gr; det(1:e,:)=false;det(end-e+1:end,:)=false;det(:,1:e)=false;det(:,end-e+1:end)=false; det(1:rMin-1,:)=false;det(rMax+1:end,:)=false;det(~range_mask,:)=false;det(:,zb-4:zb+4)=false; [dR,dD]=find(det);
            if ~isempty(dR), snr=rP(sub2ind([nR,nL],dR,dD))./(nA(sub2ind([nR,nL],dR,dD))+eps); ok=snr>10^(SNR_DB/10); if sum(ok)>MAX_D, [~,si]=sort(snr(ok),'descend');gi=find(ok);ok(gi(si(MAX_D+1:end)))=false; end; dR=dR(ok);dD=dD(ok); end
            fp=zeros(length(dR),4);nv=0;
            for i=1:length(dR)
                ri=dR(i);di=dD(i);rx32=zeros(2*nTRX,1);
                for d=1:nD, i0=(d-1)*nRX*2+1; rx32(i0:i0+nRX-1)=rc{tx0(d)}(:,ri,di); rx32(i0+nRX:i0+2*nRX-1)=rc{tx1(d)}(:,ri,di); end
                asp=fftshift(abs(fft(rx32,N_A)));[pk,pi]=max(asp); if pk<ANG_TH*median(asp),continue;end
                sA=(pi-N_A/2-1)/(N_A/2);sA=max(-1,min(1,sA));az=asind(sA);
                pd=zeros(nTRX,1);aw=zeros(nTRX,1);
                for d=1:nD, i0=(d-1)*nRX+1; t0=rc{tx0(d)}(:,ri,di);t2=rc{tx2(d)}(:,ri,di); pd(i0:i0+nRX-1)=angle(t2.*conj(t0));aw(i0:i0+nRX-1)=abs(t0); end
                aw=aw/(sum(aw)+eps);ph=sum(pd.*aw); sE=ph/pi;sE=max(-1,min(1,sE));el=asind(sE);
                rM=r_axis(ri);vM=v_axis(di); xM=rM*cosd(el)*sind(az);yM=rM*cosd(el)*cosd(az);zM=rM*sind(el);
                if zM<Z_MIN||zM>Z_MAX||abs(xM)>X_MAX||yM<Y_MIN||yM>Y_MAX,continue;end
                nv=nv+1;fp(nv,:)=[xM,yM,zM,vM];
            end
            all_pts=[all_pts; fp(1:nv,:)];
        end
        nTot=nTot+1; all_sc{nTot}=sc; all_lb(nTot)=a-1; all_pc{nTot}=all_pts;
        save(dataFile,'all_sc','all_lb','all_pc','actions','-v7.3');
        fprintf(' %df %dpts %.1fs\n',nF,size(all_pts,1),toc(tt));
    end
end
fprintf('\n===== DONE: %d scenarios =====\n',nTot);
end