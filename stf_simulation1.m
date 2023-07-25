%% Using spatiotemporal source separation to identify narrowband features in multichannel data without sinusoidal filters
%
% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.mat
%   - filterFGx.m
%   - topoplotIndie.m
% 
% mikexcohen@gmail.com

clear

%% preliminaries

% mat file containing EEG, leadfield and channel locations
load emptyEEG
EEG.srate = 512;

epochLms  = 1500; % epoch length in ms
epochLidx = round(epochLms / (1000/EEG.srate));
nTrials   = 200; % total, 1/2 per condition
EEG.pnts  = nTrials*epochLidx;
EEG.times = linspace(0,EEG.pnts/EEG.srate,EEG.pnts);


origEEG = EEG;

% normal dipoles
lf.GainN = bsxfun(@times,squeeze(lf.Gain(:,1,:)),lf.GridOrient(:,1)') + bsxfun(@times,squeeze(lf.Gain(:,2,:)),lf.GridOrient(:,2)') + bsxfun(@times,squeeze(lf.Gain(:,3,:)),lf.GridOrient(:,3)');


% indices of dipole locations
dipoleLoc1 = 109;
dipoleLoc2 = 380;

% channel sorting for plotting
[~,chansort] = sort([EEG.chanlocs.X]);

figure(1), clf
clim = [-45 45];
subplot(121), topoplotIndie(lf.GainN(:,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
title('Signal dipole projection')

subplot(122), topoplotIndie(lf.GainN(:,dipoleLoc2), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
title('Distractor dipole projection')

%% create a brain of correlated random data

% correlation matrix
cormat = rand(size(lf.GainN,2));
cormat = cormat*cormat';
cormat = .8*( cormat./max(cormat(:)) );
cormat(1:size(lf.GainN,2)+1:end) = 1;

% eigdecomp and create correlated random data
[evecs,evals] = eig(cormat);

% 1/f random data
ps   = bsxfun(@times, exp(1i*2*pi*rand(size(lf.GainN,2),floor(EEG.pnts/2))) , .1+exp(-(1:floor(EEG.pnts/2))/200) );
ps   = [ps zeros(size(lf.GainN,2),1) ps(:,end:-1:1)];
data = 500 * real(ifft(ps,[],2))'*(evecs*sqrt(evals))';

% cut into epochs
nE = floor(EEG.times(end)*EEG.srate/epochLidx); % N epochs

%% create condition wave

freq1 =  5;
freq2 = 12;

condwave = sin(2*pi* freq1 *(0:1/EEG.srate:2*(1000/freq1)/1000));
distwave = sin(2*pi* freq2 *(0:1/EEG.srate:3*(1000/freq2)/1000));

%% insert task wave into first 100 trials

epochs = reshape(data(1:nE*epochLidx,:),nE,epochLidx,size(data,2));
EEG.data = zeros(EEG.nbchan,size(epochs,2),nE);

for ti=1:nE
    
    % at random point ("nonphase-locked")
    if ti<nE/2
        st = ceil(rand*(size(epochs,2)-length(condwave)));
        epochs(ti,st:st+length(condwave)-1,dipoleLoc1) = condwave + epochs(ti,st:st+length(condwave)-1,dipoleLoc1);
    end
    
    % add distractor waveform to all trials
    st = ceil(rand*(size(epochs,2)-length(condwave)));
    epochs(ti,st:st+length(distwave)-1,dipoleLoc2) = 3*distwave;
    
    % project to scalp
    EEG.data(:,:,ti) = detrend( squeeze(epochs(ti,:,:))*lf.GainN' )';
end

[EEG.nbchan EEG.pnts EEG.trials] = size(EEG.data);
EEG.times = EEG.times(1:EEG.pnts);

%% GED for spatial filter

[cov1,cov2] = deal( zeros(EEG.nbchan) );

% covariance matrices
for ti=1:EEG.trials
    
    tdat = squeeze(EEG.data(:,:,ti));
    tdat = bsxfun(@minus,tdat,mean(tdat,2));
    if ti<nE/2
        cov1 = cov1 + (tdat*tdat')/EEG.pnts;
    else
        cov2 = cov2 + (tdat*tdat')/EEG.pnts;
    end
end

cov1 = cov1./ti;
cov2 = cov2./ti;

[evecs,evals] = eig(cov1,cov2);
maps = cov1 * evecs / (evecs' * cov1 * evecs);

% fix sign
[~,idx] = max(abs(maps(:,end)));
maps(:,end) = maps(:,end)*sign(maps(idx,end));

cdat = reshape( evecs(:,end)'*reshape(EEG.data,EEG.nbchan,[]), EEG.pnts,EEG.trials);

%% standard TF analysis on channels

frex = linspace(2,20,50);
stds = linspace(3,10,length(frex));

chan2d = reshape(EEG.data,EEG.nbchan,EEG.pnts*EEG.trials);
srce2d = reshape(epochs(:,:,dipoleLoc1)',1,[]);
comp2d = reshape(cdat,1,[]);

tf  = zeros(2,EEG.nbchan,length(frex),EEG.pnts);
stf = zeros(2,length(frex),EEG.pnts);
ctf = zeros(2,length(frex),EEG.pnts);

for fi=1:length(frex)
    
    % on channels
    as = reshape( abs( hilbert(filterFGx(chan2d,EEG.srate,frex(fi),stds(fi))')' ).^2, [EEG.nbchan EEG.pnts EEG.trials]);
    tf(1,:,fi,:) = mean(as(:,:,1:100),3);
    tf(2,:,fi,:) = mean(as(:,:,101:end),3);
    
    % on original waveform
    as = reshape(abs(hilbert(filterFGx(srce2d,EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
    stf(1,fi,:) = mean(as(:,1:100),2);
    stf(2,fi,:) = mean(as(:,101:end),2);
    
    % on GED component
    as = reshape(abs(hilbert(filterFGx(comp2d,EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
    ctf(1,fi,:) = mean(as(:,1:100),2);
    ctf(2,fi,:) = mean(as(:,101:end),2);
end

%% some plotting

figure(2), clf

fidx1 = dsearchn(frex',5);
fidx2 = dsearchn(frex',12);
tidx  = dsearchn(EEG.times',[.3 1]');
chan2plot1 = 30;
chan2plot2 = 58;


% topoplots
subplot(431), topoplotIndie(lf.GainN(:,dipoleLoc1),EEG.chanlocs);
title('Simulation')

subplot(434), topoplotIndie(squeeze(mean(tf(1,:,fidx1,tidx(1):tidx(2)),4)),EEG.chanlocs);
title('5 Hz power')

subplot(437), topoplotIndie(squeeze(mean(tf(1,:,fidx2,tidx(1):tidx(2)),4)),EEG.chanlocs);
title('12 Hz power')

subplot(4,3,10), topoplotIndie(maps(:,end),EEG.chanlocs);
title('Spatial source projection')


% TF plots
clim = [ min(stf(:)) max(stf(:)) ];
subplot(432), contourf(EEG.times,frex,squeeze(stf(1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
title('Condition A')
subplot(433), contourf(EEG.times,frex,squeeze(stf(2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
title('Condition B')

clim = [ min(tf(:)) max(tf(:)) ]*.08;
subplot(435), contourf(EEG.times,frex,squeeze(tf(1,chan2plot1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
subplot(436), contourf(EEG.times,frex,squeeze(tf(2,chan2plot1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square

subplot(438), contourf(EEG.times,frex,squeeze(tf(1,chan2plot2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
subplot(439), contourf(EEG.times,frex,squeeze(tf(2,chan2plot2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square

clim = [ min(ctf(:)) max(ctf(:)) ];
subplot(4,3,11), contourf(EEG.times,frex,squeeze(ctf(1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
subplot(4,3,12), contourf(EEG.times,frex,squeeze(ctf(2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
xlabel('Time (s)'), ylabel('Frequency (Hz)')

%%

% create time-delay matrix
ndel = 400; % number of delay embeddings (even only, please!)
padorder = [ EEG.pnts-floor(ndel/2):EEG.pnts 1:floor(ndel/2)-1 ];

[delcov1,delcov2] = deal( zeros(ndel) );

for triali=1:EEG.trials
    
    delEmb = zeros(ndel,EEG.pnts);
    for deli = 1:ndel
        delEmb(deli,:) = cdat([padorder(deli):end 1:padorder(deli)-1],triali);
    end
    
    delEmb = bsxfun(@minus,delEmb,mean(delEmb,2));
    
    if triali<nE/2
        delcov1 = delcov1 + (delEmb*delEmb')/EEG.pnts;
    else
        delcov2 = delcov2 + (delEmb*delEmb')/EEG.pnts;
    end
end

% eigendecomposition and sort matrices
[evecs,evals] = eig( delcov1,delcov2 );
[~,sidx] = sort(diag(evals));
evecs    = evecs(:,sidx);
timemap  = inv(evecs');

%%

% these maps don't find the activity; they find the filter kernel
figure(3), clf

subplot(221)
plot(1000*(0:1/EEG.srate:2*(1000/freq1)/1000)-200,condwave), set(gca,'xlim',[-400 400])
title('Simulated time series')
xlabel('Time (ms)')

subplot(222)
plot((0:ndel-1)*1000/EEG.srate,-timemap(:,end),'linew',2)
title('Empirical filter (optimal temporal basis function)')
xlabel('Time (ms)')

subplot(223)
plot(linspace(0,EEG.srate,ndel*3),abs(fft(timemap(:,end),ndel*3)),'s-','linew',2,'markerfacecolor','w','markersize',7)
set(gca,'xlim',[0 20])
title('Power spectrum of filter')
xlabel('Frequency (Hz)')

%%
