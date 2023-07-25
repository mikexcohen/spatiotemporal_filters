%% multivariate source separation

% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.mat
%   - filterFGx.m
%   - topoplotIndie.m

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

%% indices of dipole locations

dipoleLoc1 = 109;
dipoleLoc2 = 380;

% channel sorting for plotting
[~,chansort] = sort([EEG.chanlocs.X]);

figure(1), clf
clim = [-45 45];
subplot(121), topoplotIndie(lf.GainN(:,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
subplot(122), topoplotIndie(lf.GainN(:,dipoleLoc2), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');

%% create a brain of correlated random data

% correlation matrix
% cormat = rand(size(lf.GainN,2));
% cormat = cormat*cormat';
% cormat = .8*( cormat./max(cormat(:)) );
% cormat(1:size(lf.GainN,2)+1:end) = 1;
% 
% % eigdecomp and create correlated random data
% [evecs,evals] = eig(cormat);
% 
% % 1/f random data
% ps   = bsxfun(@times, exp(1i*2*pi*rand(size(lf.GainN,2),floor(EEG.pnts/2))) , .1+exp(-(1:floor(EEG.pnts/2))/200) );
% ps   = [ps zeros(size(lf.GainN,2),1) ps(:,end:-1:1)];
% data = 500 * real(ifft(ps,[],2))'*(evecs*sqrt(evals))';

data = .3*randn(EEG.pnts,size(lf.GainN,2));

% cut into epochs
nE = floor(EEG.times(end)*EEG.srate/epochLidx); % N epochs

%% create condition wave

freq1 =  5;
freq2 = 12;

condwave1 = sin(2*pi*freq1*(0:1/EEG.srate:3*(1000/freq1)/1000));
condwave2 = sin(2*pi*freq2*(0:1/EEG.srate:4*(1000/freq2)/1000));

%% insert task wave into first 100 trials

epochs = reshape(data(1:nE*epochLidx,:),nE,epochLidx,size(data,2));
EEG.data = zeros(EEG.nbchan,size(epochs,2),nE);

times2start1 = dsearchn(EEG.times',.4):dsearchn(EEG.times',.55);
times2start2 = dsearchn(EEG.times',.5):dsearchn(EEG.times',.65);

for ti=1:nE
    
    if ti<nE/2+1
        % at random point ("nonphase-locked")
        st = times2start1(ceil(rand*length(times2start1)));
        epochs(ti,st:st+length(condwave1)-1,dipoleLoc1) = condwave1 + epochs(ti,st:st+length(condwave1)-1,dipoleLoc1);
        
        st = times2start2(ceil(rand*length(times2start2)));
        epochs(ti,st:st+length(condwave2)-1,dipoleLoc2) = condwave2 + epochs(ti,st:st+length(condwave2)-1,dipoleLoc2);
    end
    
    % project to scalp
    EEG.data(:,:,ti) = detrend( squeeze(epochs(ti,:,:))*lf.GainN' )';
end

[EEG.nbchan EEG.pnts EEG.trials] = size(EEG.data);
EEG.times = EEG.times(1:EEG.pnts);

%% GED for spatial filter

[cov1,cov2] = deal( zeros(EEG.nbchan) );

% covariance matrices per trial
for ti=1:EEG.trials
    
    tdat = squeeze(EEG.data(:,:,ti));
    tdat = bsxfun(@minus,tdat,mean(tdat,2));
    
    % add to cov1 or cov2
    if ti<nE/2+1
        cov1 = cov1 + (tdat*tdat')/EEG.pnts;
    else
        cov2 = cov2 + (tdat*tdat')/EEG.pnts;
    end
end

cov1 = cov1./ti;
cov2 = cov2./ti;

% GED
[evecs,evals] = eig(cov1,cov2);
[~,sidx] = sort(diag(evals),'descend');
evals = diag(evals);
evals = evals(sidx);
evecs = evecs(:,sidx);
maps  = cov1 * evecs;

% fix sign
[~,idx] = max(abs(maps(:,1)));
maps(:,1) = maps(:,1)*sign(maps(idx,1));
maps(:,2) = maps(:,2)*sign(maps(idx,1));

% compute component time series (projections)
cdat = reshape( evecs(:,1:2)'*reshape(EEG.data,EEG.nbchan,[]), [ 2 EEG.pnts EEG.trials ]);

%% standard TF analysis on channels

% pick component 1 or 2 to analyze and plot in the next section
whichComp2plot = 2;

frex = linspace(2,20,30);
stds = linspace(3,10,length(frex));

chan2d = reshape(EEG.data,EEG.nbchan,EEG.pnts*EEG.trials);
comp2d = reshape(cdat,2,[]);
clear srce2d
srce2d(1,:) = reshape(epochs(:,:,dipoleLoc1)',1,[]);
srce2d(2,:) = reshape(epochs(:,:,dipoleLoc2)',1,[]);

tf  = zeros(2,EEG.nbchan,length(frex),EEG.pnts);
stf = zeros(2,length(frex),EEG.pnts);
ctf = zeros(2,length(frex),EEG.pnts);

for fi=1:length(frex)
    
    % on channels
    as = reshape( abs( hilbert(filterFGx(chan2d,EEG.srate,frex(fi),stds(fi))')' ).^2, [EEG.nbchan EEG.pnts EEG.trials]);
    tf(1,:,fi,:) = mean(as(:,:,1:100),3);
    tf(2,:,fi,:) = mean(as(:,:,101:end),3);
    
    % on original waveform
    as = reshape(abs(hilbert(filterFGx(srce2d(whichComp2plot,:),EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
    stf(1,fi,:) = mean(as(:,1:100),2);
    stf(2,fi,:) = mean(as(:,101:end),2);
    
    % on GED components
    as = reshape(abs(hilbert(filterFGx(comp2d(whichComp2plot,:),EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
    ctf(1,fi,:) = mean(as(:,1:100),2);
    ctf(2,fi,:) = mean(as(:,101:end),2);
end

%% some plotting

figure(10+whichComp2plot), clf

fidx1 = dsearchn(frex',5);
fidx2 = dsearchn(frex',12);
tidx  = dsearchn(EEG.times',[.3 1]');
chan2plot1 = 30;
chan2plot2 = 58;


% topoplots
dips=[dipoleLoc1 dipoleLoc2];
subplot(431), topoplotIndie(lf.GainN(:,dips(whichComp2plot)),EEG.chanlocs);
title('True source')

subplot(434), topoplotIndie(squeeze(mean(tf(1,:,fidx1,tidx(1):tidx(2)),4)),EEG.chanlocs,'emarker2',{chan2plot1,'^','k',3});
title('Electrode')

subplot(437), topoplotIndie(squeeze(mean(tf(1,:,fidx2,tidx(1):tidx(2)),4)),EEG.chanlocs,'emarker2',{chan2plot2,'^','k',3});
title('Electrode')

subplot(4,3,10), topoplotIndie(maps(:,whichComp2plot),EEG.chanlocs);
title('Component')

% TF plots
clim = [ min(stf(:)) max(stf(:)) ];
subplot(432), contourf(EEG.times,frex,squeeze(stf(1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
title('Wave condition')
subplot(433), contourf(EEG.times,frex,squeeze(stf(2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
title('Control condition')

clim = [ min(tf(:)) max(tf(:)) ];
subplot(435), contourf(EEG.times,frex,squeeze(tf(1,chan2plot1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
subplot(436), contourf(EEG.times,frex,squeeze(tf(2,chan2plot1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square

subplot(438), contourf(EEG.times,frex,squeeze(tf(1,chan2plot2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
subplot(439), contourf(EEG.times,frex,squeeze(tf(2,chan2plot2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square

clim = [ min(ctf(:)) max(ctf(:)) ];
subplot(4,3,11), contourf(EEG.times,frex,squeeze(ctf(1,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square
xlabel('Time (s)'), ylabel('Frequency (Hz)')
subplot(4,3,12), contourf(EEG.times,frex,squeeze(ctf(2,:,:)),40,'linecolor','none'), set(gca,'clim',clim), axis square

%% end
