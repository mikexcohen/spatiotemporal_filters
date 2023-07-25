%% multivariate source separation

% You will need the following files in the current directory or Matlab path:
%   - emptyEEG.ma

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
dipoleLoc2 = 135;

% channel sorting for plotting
[~,chansort] = sort([EEG.chanlocs.X]);

figure(1), clf
clim = [-45 45];
subplot(231)
topoplotIndie(lf.GainN(:,dipoleLoc1), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
title('Simulation dipole 1')

subplot(232)
topoplotIndie(lf.GainN(:,dipoleLoc2), EEG.chanlocs,'maplimits',clim,'numcontour',0,'electrodes','numbers','shading','interp');
title('Simulation dipole 2')

%% create a brain of correlated random data
data = .1*randn(EEG.pnts,size(lf.GainN,2));

% cut into epochs
nE = floor(EEG.times(end)*EEG.srate/epochLidx); % N epochs

%% create condition wave

freq1 = 12;
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
maps(:,1) = zscore( maps(:,1)*sign(maps(idx,1)) );
[~,idx] = max(abs(maps(:,2)));
maps(:,2) = zscore( maps(:,2)*sign(maps(idx,2)) );

% compute component time series (projections)
cdat = reshape( evecs(:,1:2)'*reshape(EEG.data,EEG.nbchan,[]), [ 2 EEG.pnts EEG.trials ]);

figure(1)
subplot(234), topoplotIndie(maps(:,1),EEG.chanlocs); title('Component 1')
subplot(235), topoplotIndie(maps(:,2),EEG.chanlocs); title('Component 2')

subplot(236), plot(evals,'s-','linew',2,'markersize',10,'markerfacecolor','k')
set(gca,'xlim',[0 15])
ylabel('\lambda'), title('Eigenvalues of decomposition'), axis square

%% factor analysis

% factoran using no special inputs and assuming 2 factors
[Lambda, Psi, T, stats, F] = factoran(reshape(EEG.data(:,:,1:100),EEG.nbchan,[])',2);

%% standard TF analysis on channels

frex = linspace(2,20,20);
stds = linspace(3,10,length(frex));

chan2d = reshape(EEG.data,EEG.nbchan,EEG.pnts*EEG.trials);
comp2d = reshape(cdat,2,[]);
clear srce2d
srce2d(1,:) = reshape(epochs(:,:,dipoleLoc1)',1,[]);
srce2d(2,:) = reshape(epochs(:,:,dipoleLoc2)',1,[]);

tf  = zeros(2,EEG.nbchan,length(frex),EEG.pnts); % channel TF (not currently used)
stf = zeros(2,2,length(frex),EEG.pnts); % source (dipole) TF
ctf = zeros(2,2,length(frex),EEG.pnts); % component TF
ftf = zeros(2,length(frex),EEG.pnts); % factor TF

for fi=1:length(frex)
    
    % on channels
%     as = reshape( abs( hilbert(filterFGx(chan2d,EEG.srate,frex(fi),stds(fi))')' ).^2, [EEG.nbchan EEG.pnts EEG.trials]);
%     tf(1,:,fi,:) = mean(as(:,:,1:100),3);
%     tf(2,:,fi,:) = mean(as(:,:,101:end),3);
    
    for compi=1:2
        % on original waveform
        as = reshape(abs(hilbert(filterFGx(srce2d(compi,:),EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
        stf(compi,1,fi,:) = mean(as(:,1:100),2);
        stf(compi,2,fi,:) = mean(as(:,101:end),2);
        
        % on GED component
        as = reshape(abs(hilbert(filterFGx(comp2d(compi,:),EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,EEG.trials);
        ctf(compi,1,fi,:) = mean(as(:,1:100),2);
        ctf(compi,2,fi,:) = mean(as(:,101:end),2);
        
        
        % on factor component scores
        as = reshape(abs(hilbert(filterFGx(F(:,compi)',EEG.srate,frex(fi),stds(fi)))).^2 ,EEG.pnts,nE/2);
        ftf(compi,fi,:) = mean(as,2);
        
    end
end

%% some plotting

figure(2), clf

fidx1 = dsearchn(frex',5);
fidx2 = dsearchn(frex',12);
tidx  = dsearchn(EEG.times',[.3 1]');

for compi=1:2
    for condi=1:2
        
        % source
        subplot(2,4,compi*2+condi-2)
        contourf(EEG.times,frex,squeeze(stf(compi,condi,:,:)),40,'linecolor','none')
        tmp = reshape(stf(compi,:,:,:),1,[]);
        set(gca,'clim',[ min(tmp) max(tmp) ]), axis square
        title([ 'Dipole ' num2str(compi) ])
        
        % component
        subplot(2,4,compi*2+condi+2)
        contourf(EEG.times,frex,squeeze(ctf(compi,condi,:,:)),40,'linecolor','none')
        tmp = reshape(ctf(compi,1,:,:),1,[]);
        set(gca,'clim',[ min(tmp) max(tmp) ]), axis square
        title([ 'Component ' num2str(compi) ])
        xlabel('Time (s)'), ylabel('Frequency (Hz)')
    end
end

%% factoran plotting

figure(3), clf

for compi=1:2
    
    % topomaps
    subplot(2,2,compi)
    topoplotIndie(Lambda(:,compi),EEG.chanlocs);
    
    % TF plots
    subplot(2,2,compi+2)
    contourf(EEG.times,frex,squeeze(ftf(compi,:,:)),40,'linecolor','none')
    tmp = reshape(ftf(compi,:,:),1,[]);
    set(gca,'clim',[ min(tmp) max(tmp) ]), axis square
    title([ 'Factor ' num2str(compi) ])
    xlabel('Time (s)'), ylabel('Frequency (Hz)')
end

%% end
