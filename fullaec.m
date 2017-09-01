clc;
% Partitioned block frequency domain adaptive filtering NLMS and
% standard time-domain sample-based NLMS
%near is micphone captured signal
fid=fopen('near_16kHz_32f.pcm', 'rb'); % Load far end
ssin=fread(fid,inf,'float32');
fclose(fid);
%far is speaker played music
fid=fopen('far_16KHz_32f.pcm', 'rb'); % Load fnear end
rrin=fread(fid,inf,'float32');
fclose(fid);

rand('state',13);
fs=16000;
mult=fs/8000;
if fs == 8000
cohRange = 2:3;
elseif fs==16000
cohRange = 2;
end

% Flags
NLPon=1; % NLP on
CNon=0; % Comfort noise on
PLTon=0; % Plotting on

M = 16; % Number of partitions
N = 64; % Partition length
L = M*N; % Filter length

mufb = 1.5;

VADtd=48;
alp = 0.15; % Power estimation factor 
alc = 0.1; % Coherence estimation factor
beta = 0.9; % Plotting factor
%% Changed a little %%
step = 0.1875;%0.1875; % Downward step size
%%
threshold=0.5e-3;
echoBandRange = ceil(50/fs*N):floor(3000*2/fs*N);

suppState = 1;
transCtr = 0;

Nt=1;
vt=1;

ramp = 1.0003; % Upward ramp
rampd = 0.999; % Downward ramp
cvt = 20; % Subband VAD threshold;
nnthres = 20; % Noise threshold

shh=logspace(-1.3,-2.2,N+1)';
shh=ones(1,N+1)';
sh=[shh;flipud(shh(2:end-1))]; % Suppression profile

len=length(ssin);
w=zeros(L,1); % Sample-based TD(time domain) NLMS
WFb=zeros(N+1,M); % Block-based FD(frequency domain) NLMS
WFbOld=zeros(N+1,M); % Block-based FD NLMS
YFb=zeros(N+1,M);
erfb=zeros(len,1);
erfb3=zeros(len,1);

ercn=zeros(len,1);
zm=zeros(N,1);
XFm=zeros(N+1,M);
YFm=zeros(N+1,M);
pn0=10*ones(N+1,1);
pn=zeros(N+1,1);
NN=len;
Nb=floor(NN/N)-M;
erifb=zeros(Nb+1,1)+0.1;
erifb3=zeros(Nb+1,1)+0.1;
ericn=zeros(Nb+1,1)+0.1;
dri=zeros(Nb+1,1)+0.1;
start=1;
xo=zeros(N,1);
do=xo;
eo=xo;

echoBands=zeros(Nb+1,1);
cohxdAvg=zeros(Nb+1,1);
cohxdSlow=zeros(Nb+1,N+1);
cohedSlow=zeros(Nb+1,N+1);
%overdriveM=zeros(Nb+1,N+1);
cohxdFastAvg=zeros(Nb+1,1);
cohxdAvgBad=zeros(Nb+1,1);
cohedAvg=zeros(Nb+1,1);
cohedFastAvg=zeros(Nb+1,1);
hnledAvg=zeros(Nb+1,1);
hnlxdAvg=zeros(Nb+1,1);
ovrdV=zeros(Nb+1,1);
dIdxV=zeros(Nb+1,1);
SLxV=zeros(Nb+1,1);
hnlSortQV=zeros(Nb+1,1);
hnlPrefAvgV=zeros(Nb+1,1);
mutInfAvg=zeros(Nb+1,1);
%overdrive=zeros(Nb+1,1);
hnled = zeros(N+1, 1);
weight=zeros(N+1,1);
hnlMax = zeros(N+1, 1);
hnl = zeros(N+1, 1);
overdrive = ones(1, N+1);
xfwm=zeros(N+1,M);
dfm=zeros(N+1,M);
WFbD=ones(N+1,1);

fbSupp = 0;
hnlLocalMin = 1;
cohxdLocalMin = 1;
hnlLocalMinV=zeros(Nb+1,1);
cohxdLocalMinV=zeros(Nb+1,1);
hnlMinV=zeros(Nb+1,1);
dkEnV=zeros(Nb+1,1);
ekEnV=zeros(Nb+1,1);
ovrd = 2;
ovrdPos = floor((N+1)/4);
ovrdSm = 2;
hnlMin = 1;
minCtr = 0;
SeMin = 0;
SdMin = 0;
SeLocalAvg = 0;
SeMinSm = 0;
divergeFact = 1;
dIdx = 1;
hnlMinCtr = 0;
hnlNewMin = 0;
divergeState = 0;

Sy=ones(N+1,1);
Sym=1e7*ones(N+1,1);

wins=[0;sqrt(hanning(2*N-1))];
ubufn=zeros(2*N,1);
ebuf=zeros(2*N,1);
ebuf2=zeros(2*N,1);
ebuf4=zeros(2*N,1);
mbuf=zeros(2*N,1);

cohedFast = zeros(N+1,1);
cohxdFast = zeros(N+1,1);
cohxd = zeros(N+1,1);
Se = zeros(N+1,1);
Sd = zeros(N+1,1);
Sx = zeros(N+1,1);
SxBad = zeros(N+1,1);
Sed = zeros(N+1,1);
Sxd = zeros(N+1,1);
SxdBad = zeros(N+1,1);
hnledp=[];

cohxdMax = 0;
for kk=1:Nb
    pos = N * (kk-1) + start;  
    %far is speaker played music
    xk = rrin(pos:pos+N-1);
    %near is micphone captured signal
    dk = ssin(pos:pos+N-1);
    
    %far end signal process
    xx = [xo;xk];
    xo = xk;
    tmp = fft(xx);
    XX = tmp(1:N+1);%this is overlap save need, end half needed(because fftshift not used)  
    %near end signal process
    dd = [do;dk]; % Overlap
    do = dk;
    tmp = fft(dd); % Frequency domain
    DD = tmp(1:N+1);
    
    %far end Power estimation
    pn0 = (1 - alp) * pn0 + alp * real(XX.* conj(XX));
    pn = pn0;    
    
    %-------------Filtering
    XFm(:,1) = XX;
    for mm=0:(M-1)
        m=mm+1;
        YFb(:,m) = XFm(:,m) .* WFb(:,m);
    end
    yfk = sum(YFb,2);
    tmp = [yfk ; flipud(conj(yfk(2:N)))];
    ykt = real(ifft(tmp));
    ykfb = ykt(end-N+1:end);
    
    % --------Error estimation
    ekfb = dk - ykfb;
    %For robustness
%     if sum(abs(ekfb)) < sum(abs(dk))
%         ekfb = dk - ykfb;
%     erfb(pos:pos+N-1) = ekfb;
%     else
%         ekfb = dk;
%     erfb(pos:pos+N-1) = dk;
%     end

    erfb(pos:pos+N-1) = ekfb;
    tmp = fft([zm;ekfb]); % FD version for cancelling part (overlap-save)
    Ek = tmp(1:N+1);

    % ------------------------ Adaptation
%   Ek2 = Ek ./(M*pn + 0.001); % Normalized error
    Ek2 = Ek ./(pn + 0.001); % Normalized error
    
    absEf = max(abs(Ek2), threshold);
    absEf = ones(N+1,1)*threshold./absEf;
    Ek2 = Ek2.*absEf;

    mEk = mufb.*Ek2;
    PP = conj(XFm).*(ones(M,1) * mEk')';
    tmp = [PP ; flipud(conj(PP(2:N,:)))];
    IFPP = real(ifft(tmp));
    PH = IFPP(1:N,:);
    tmp = fft([PH;zeros(N,M)]);
    FPH = tmp(1:N+1,:);
    WFb = WFb + FPH;

%     if mod(kk, 10*mult) == 0
        WFbEn = sum(real(WFb.*conj(WFb)));
        %WFbEn = sum(abs(WFb));
        [tmp, dIdx] = max(WFbEn);

        WFbD = sum(abs(WFb(:, dIdx)),2);
        %WFbD = WFbD / (mean(WFbD) + 1e-10);
        WFbD = min(max(WFbD, 0.5), 4);
%     end
    dIdxV(kk) = dIdx;
    
    dIdx = 2;
    
    % NLP
    if (NLPon)  
        ee = [eo;ekfb];
        eo = ekfb;
        window = wins;
        gamma = 0.93;
        tmp = fft(xx.*window);
        xf = tmp(1:N+1);
        tmp = fft(dd.*window);
        df = tmp(1:N+1);
        tmp = fft(ee.*window);
        ef = tmp(1:N+1);

        xfwm(:,1) = xf;
        xf = xfwm(:,dIdx);
     
        dfm(:,1) = df;
        
        SxOld = Sx;

        Se = gamma*Se + (1-gamma)*real(ef.*conj(ef));
        Sd = gamma*Sd + (1-gamma)*real(df.*conj(df));
        Sx = gamma*Sx + (1 - gamma)*real(xf.*conj(xf));

        %xRatio = real(xfwm(:,1).*conj(xfwm(:,1))) ./ ...
        % (real(xfwm(:,2).*conj(xfwm(:,2))) + 1e-10);
        %xRatio = Sx ./ (SxOld + 1e-10);
        %SLx = log(1/(N+1)*sum(xRatio)) - 1/(N+1)*sum(log(xRatio));
        %SLxV(kk) = SLx;

%         freqSm = 0.9;
%         Sx = filter(freqSm, [1 -(1-freqSm)], Sx);
%         Sx(end:1) = filter(freqSm, [1 -(1-freqSm)], Sx(end:1));
%         Se = filter(freqSm, [1 -(1-freqSm)], Se);
%         Se(end:1) = filter(freqSm, [1 -(1-freqSm)], Se(end:1));
%         Sd = filter(freqSm, [1 -(1-freqSm)], Sd);
%         Sd(end:1) = filter(freqSm, [1 -(1-freqSm)], Sd(end:1));

        %SeFast = ef.*conj(ef);
        %SdFast = df.*conj(df);
        %SxFast = xf.*conj(xf);
        %cohedFast = 0.9*cohedFast + 0.1*SeFast ./ (SdFast + 1e-10);
        %cohedFast(find(cohedFast > 1)) = 1;
        %cohedFast(find(cohedFast > 1)) = 1 ./ cohedFast(find(cohedFast>1));
        %cohedFastAvg(kk) = mean(cohedFast(echoBandRange));
        %cohedFastAvg(kk) = min(cohedFast);

        %cohxdFast = 0.8*cohxdFast + 0.2*log(SdFast ./ (SxFast + 1e-10));
        %cohxdFastAvg(kk) = mean(cohxdFast(echoBandRange));

        % coherence
        Sxd = gamma*Sxd + (1 - gamma)*xf.*conj(df);
        Sed = gamma*Sed + (1-gamma)*ef.*conj(df);

        cohed = real(Sed.*conj(Sed))./(Se.*Sd + 1e-10);
        cohedAvg(kk) = mean(cohed(echoBandRange));
        cohxd = real(Sxd.*conj(Sxd))./(Sx.*Sd + 1e-10);
        freqSm = 0.55;
        cohxd(2:end) = filter(freqSm, [1 -(1-freqSm)], cohxd(2:end));
        cohxd(end:2) = filter(freqSm, [1 -(1-freqSm)], cohxd(end:2));
        cohxdAvg(kk) = mean(cohxd(echoBandRange));
        %cohxdAvg(kk) = (cohxd(32));
        %cohxdAvg(kk) = max(cohxd);

        %xf = xfm(:,dIdx);
        %SxBad = gamma*SxBad + (1 - gamma)*real(xf.*conj(xf));
        %SxdBad = gamma*SxdBad + (1 - gamma)*xf.*conj(df);
        %cohxdBad = real(SxdBad.*conj(SxdBad))./(SxBad.*Sd + 0.01);
        %cohxdAvgBad(kk) = mean(cohxdBad);

        %for j=1:N+1
        % mutInf(j) = 0.9*mutInf(j) + 0.1*information(abs(xfm(j,:)), abs(dfm(j,:)));
        %end
        %mutInfAvg(kk) = mean(mutInf);

        %hnled = cohedFast;
        %xIdx = find(cohxd > 1 - cohed);
        %hnled(xIdx) = 1 - cohxd(xIdx);
        %hnled = 1 - max(cohxd, 1-cohedFast);
        hnled = min(1 - cohxd, cohed);
        %hnled = 1 - cohxd;
        %hnled = max(1 - (cohxd + (1-cohedFast)), 0);
        %hnled = 1 - max(cohxd, 1-cohed);

        if kk > 1
            cohxdSlow(kk,:) = 0.99*cohxdSlow(kk-1,:) + 0.01*cohxd';
            cohedSlow(kk,:) = 0.99*cohedSlow(kk-1,:) + 0.01*(1-cohed)';
        end


        if 0
        %if kk > 50
            %idx = find(hnled > 0.3);
            hnlMax = hnlMax*0.9999;
            %hnlMax(idx) = max(hnlMax(idx), hnled(idx));
            hnlMax = max(hnlMax, hnled);
            %overdrive(idx) = max(log(hnlMax(idx))/log(0.99), 1);
            avgHnl = mean(hnlMax(echoBandRange));
            if avgHnl > 0.3
                overdrive = max(log(avgHnl)/log(0.99), 1);
            end
            weight(4:end) = max(hnlMax) - hnlMax(4:end);
        end
        
        

        %[hg, gidx] = max(hnled);
        %fnrg = Sx(gidx) / (Sd(gidx) + 1e-10);
        
        %[tmp, bidx] = find((Sx / Sd + 1e-10) > fnrg);
        %hnled(bidx) = hg;


        %cohed1 = mean(cohed(cohRange)); % range depends on bandwidth
        %cohed1 = cohed1^2;
        %echoBands(kk) = length(find(cohed(echoBandRange) < 0.25))/length(echoBandRange);

        %if (fbSupp == 0)
        % if (echoBands(kk) > 0.8)
        % fbSupp = 1;
        % end
        %else
        % if (echoBands(kk) < 0.6)
        % fbSupp = 0;
        % end
        %end
        %overdrive(kk) = 7.5*echoBands(kk) + 0.5;
        
% Factor by which to weight other bands
%if (cohed1 < 0.1)
% w = 0.8 - cohed1*10*0.4;
%else
% w = 0.4;
%end

% Weight coherence subbands
%hnled = w*cohed1 + (1 - w)*cohed;
%hnled = (hnled).^2;
%cohed(floor(N/2):end) = cohed(floor(N/2):end).^2;
        %if fbSupp == 1
        % cohed = zeros(size(cohed));
        %end
        %cohed = cohed.^overdrive(kk);

        %hnled = gamma*hnled + (1 - gamma)*cohed;
% Additional hf suppression
%hnledp = [hnledp ; mean(hnled)];
%hnled(floor(N/2):end) = hnled(floor(N/2):end).^2;
%ef = ef.*((weight*(min(1 - hnled)).^2 + (1 - weight).*(1 - hnled)).^2);

        cohedMean = mean(cohed(echoBandRange));
        %aggrFact = 4*(1-mean(hnled())) + 1;
        %[hnlSort, hnlSortIdx] = sort(hnled(ecechoBandRangehoBandRange));
        [hnlSort, 	hnlSortIdx] = sort(1-cohxd(echoBandRange));
        [xSort, xSortIdx] = sort(Sx);
        %aggrFact = (1-mean(hnled(echoBandRange)));
        %hnlSortQ = hnlSort(qIdx);
        hnlSortQ = mean(1 - cohxd(echoBandRange));
        %hnlSortQ = mean(1 - cohxd);

        [hnlSort2, hnlSortIdx2] = sort(hnled(echoBandRange));
        %[hnlSort2, hnlSortIdx2] = sort(hnled);
        hnlQuant = 0.75;
        hnlQuantLow = 0.5;
        qIdx = floor(hnlQuant*length(hnlSort2));
        qIdxLow = floor(hnlQuantLow*length(hnlSort2));
        hnlPrefAvg = hnlSort2(qIdx);
        hnlPrefAvgLow = hnlSort2(qIdxLow);

        suppState = 1;
        if hnlSortQ < cohxdLocalMin & hnlSortQ < 0.75
            cohxdLocalMin = hnlSortQ;
        end

        if suppState == 0
            hnled = cohed;
            hnlPrefAvg = cohedMean;
            hnlPrefAvgLow = cohedMean;
        end

        %if hnlPrefAvg < hnlLocalMin & hnlPrefAvg < 0.6
        if hnlPrefAvgLow < hnlLocalMin & hnlPrefAvgLow < 0.6
            %hnlLocalMin = hnlPrefAvg;
            %hnlMin = hnlPrefAvg;
            hnlLocalMin = hnlPrefAvgLow;
            hnlMin = hnlPrefAvgLow;
            hnlNewMin = 1;
            hnlMinCtr = 0;
            if hnlMinCtr == 0
                hnlMinCtr = hnlMinCtr + 1;
            else
                hnlMinCtr = 0;
                hnlMin = hnlLocalMin;
                SeLocalMin = SeQ;
                SdLocalMin = SdQ;
                SeLocalAvg = 0;
                minCtr = 0;
                ovrd = max(log(0.0001)/log(hnlMin), 2);
                divergeFact = hnlLocalMin;
            end
        end

        if hnlNewMin == 1
            hnlMinCtr = hnlMinCtr + 1;
        end
        if hnlMinCtr == 2
            hnlNewMin = 0;
            hnlMinCtr = 0;
            %ovrd = max(log(0.0001)/log(hnlMin), 2);
%             ovrd = max(log(0.0001)/(log(hnlMin + 1e-10) + 1e-10), 5);
%           ovrd = max(log(0.00001)/(log(hnlMin + 1e-10) + 1e-10), 5);
            %ovrd = max(log(0.0001)/log(hnlPrefAvg), 2);
          ovrd = max(log(0.001)/log(hnlMin), 8);
        end
        hnlLocalMin = min(hnlLocalMin + 0.0008/mult, 1);
        cohxdLocalMin = min(cohxdLocalMin + 0.0004/mult, 1);

        if ovrd < ovrdSm
            ovrdSm = 0.99*ovrdSm + 0.01*ovrd;
        else
            ovrdSm = 0.9*ovrdSm + 0.1*ovrd;
        end

        ekEn = sum(Se);
        dkEn = sum(Sd);

        if divergeState == 0
            if ekEn > dkEn
                ef = df;
                divergeState = 1;
            end
        else
            if ekEn*1.05 < dkEn
                divergeState = 0;
            else
                ef = df;
            end
        end

        if ekEn > dkEn*19.95
            WFb=zeros(N+1,M); % Block-based FD NLMS
        end

        ekEnV(kk) = ekEn;
        dkEnV(kk) = dkEn;

        hnlLocalMinV(kk) = hnlLocalMin;
        cohxdLocalMinV(kk) = cohxdLocalMin;
        hnlMinV(kk) = hnlMin;
        %cohxdMaxLocal = max(cohxdSlow(kk,:));
        %if kk > 50
        %cohxdMaxLocal = 1-hnlSortQ;
        %if cohxdMaxLocal > 0.5
        % %if cohxdMaxLocal > cohxdMax
        % odScale = max(log(cohxdMaxLocal)/log(0.95), 1);
        % %overdrive(7:end) = max(log(cohxdSlow(kk,7:end))/log(0.9), 1);
        % cohxdMax = cohxdMaxLocal;
        % end
        %end
        %end
        %cohxdMax = cohxdMax*0.999;

        %overdriveM(kk,:) = max(overdrive, 1);
        %aggrFact = 0.25;
        aggrFact = 0.3;
        %aggrFact = 0.5*propLow;
        %if fs == 8000
        % wCurve = [0 ; 0 ; aggrFact*sqrt(linspace(0,1,N-1))' + 0.1];
        %else
        % wCurve = [0; 0; 0; aggrFact*sqrt(linspace(0,1,N-2))' + 0.1];
        %end
        wCurve = [0; aggrFact*sqrt(linspace(0,1,N))' + 0.1];
        % For sync with C
        %if fs == 8000
        % wCurve = wCurve(2:end);
        %else
        % wCurve = wCurve(1:end-1);
        %end
        %weight = aggrFact*(sqrt(linspace(0,1,N+1)'));
        %weight = aggrFact*wCurve;
        weight = wCurve;
        %weight = aggrFact*ones(N+1,1);
        %weight = zeros(N+1,1);
        %hnled = weight.*min(hnled) + (1 - weight).*hnled;
        %hnled = weight.*min(mean(hnled(echoBandRange)), hnled) + (1 - weight).*hnled;
        %hnled = weight.*min(hnlSortQ, hnled) + (1 - weight).*hnled;

        %hnlSortQV(kk) = mean(hnled);
        %hnlPrefAvgV(kk) = mean(hnled(echoBandRange));

        hnled = weight.*min(hnlPrefAvg, hnled) + (1 - weight).*hnled;

        %od = aggrFact*(sqrt(linspace(0,1,N+1)') + aggrTerm);
        %od = 4*(sqrt(linspace(0,1,N+1)') + 1/4);

%       ovrdFact = (ovrdSm - 1) / sqrt(ovrdPos/(N+1));
%       ovrdFact = ovrdSm / sqrt(echoBandRange(floor(length(echoBandRange)/2))/(N+1));
%       od = ovrdFact*sqrt(linspace(0,1,N+1))' + 1;
        %od = ovrdSm*ones(N+1,1).*abs(WFb(:,dIdx))/(max(abs(WFb(:,dIdx)))+1e-10);

        %od = ovrdSm*ones(N+1,1);
%        od = ovrdSm*WFbD.*(sqrt(linspace(0,1,N+1))' + 1);

         od = ovrdSm*fliplr(sqrt(linspace(0,1,N+1))' + 1);
        %od = 4*(sqrt(linspace(0,1,N+1))' + 1);

        %od = 2*ones(N+1,1);
        %od = 2*ones(N+1,1);
        %sshift = ((1-hnled)*2-1).^3+1;
        sshift = 1.5*ones(N+1,1);

        hnled = hnled.^(od.*sshift);

        %if hnlg > 0.75
            %if (suppState ~= 0)
            % transCtr = 0;
            %end
        % suppState = 0;
        %elseif hnlg < 0.6 & hnlg > 0.2
        % suppState = 1;
        %elseif hnlg < 0.1
            %hnled = zeros(N+1, 1);
            %if (suppState ~= 2)
            % transCtr = 0;
            %end
        % suppState = 2;
        %else
        % if (suppState ~= 2)
        % transCtr = 0;
        % end
        % suppState = 2;
        %end
        %if suppState == 0
        % hnled = ones(N+1, 1);
        %elseif suppState == 2
        % hnled = zeros(N+1, 1);
        %end
        %hnled(find(hnled < 0.1)) = 0;
        %hnled = hnled.^2;
        %if transCtr < 5
            %hnl = 0.75*hnl + 0.25*hnled;
        % transCtr = transCtr + 1;
        %else
            hnl = hnled;
        %end
        %hnled(find(hnled < 0.05)) = 0;
        ef = ef.*(hnl);

        ef = ef.*(min(1 - cohxd, cohed).^2);
%         ef = ef.*((1-cohxd).^2);
        
        ovrdV(kk) = ovrdSm;
        %ovrdV(kk) = dIdx;
        %ovrdV(kk) = divergeFact;
        %hnledAvg(kk) = 1-mean(1-cohedFast(echoBandRange));
        hnledAvg(kk) = 1-mean(1-cohed(echoBandRange));
        hnlxdAvg(kk) = 1-mean(cohxd(echoBandRange));
        %hnlxdAvg(kk) = cohxd(5);
        %hnlSortQV(kk) = mean(hnled);
        hnlSortQV(kk) = hnlPrefAvgLow;
        hnlPrefAvgV(kk) = hnlPrefAvg;
        %hnlAvg(kk) = propLow;
        %ef(N/2:end) = 0;
        %ner = (sum(Sd) ./ (sum(Se.*(hnl.^2)) + 1e-10));

        Fmix = ef;

    % Overlap and add in time domain for smoothness
    tmp = [Fmix ; flipud(conj(Fmix(2:N)))];
    mixw = wins.*real(ifft(tmp));
    mola = mbuf(end-N+1:end) + mixw(1:N);
    mbuf = mixw;
    ercn(pos:pos+N-1) = mola;%%%%%----you can hear the effect by sound(100*ercn,16000),add by Shichaog
    end % NLPon


    % Filter update
   Ek2 = Ek ./(100*pn + 0.001); % Normalized error


% Shift old FFTs
    XFm(:,2:end) = XFm(:,1:end-1);
    YFm(:,2:end) = YFm(:,1:end-1);
    xfwm(:,2:end) = xfwm(:,1:end-1);
    dfm(:,2:end) = dfm(:,1:end-1);


end


figure(9);
    subplot(2,2,1);plot(ssin);
    subplot(2,2,2);plot(rrin);
    subplot(2,2,3);plot(100*ercn);


hold off
