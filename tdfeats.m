
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TDFEATS	Compute four time domain features
%   1. mav: mean absolute value			
%   2. len: waveform length
%   3. zero_count: number of zero crossings
%   4. turns: number of slope sign changes
% 
%     Inputs:
%         DataSet: matrix of the raw EMG data (signal x channel)
%         Nframe: total number of windows in DataSet. We usually process
%         windows one by one, so Nframe is usually set to 1
%         WL: window length
%         WInc: window increment
%     Outputs:
%         Feature_matrix: the output feature matrix (feature (mave, len, zero_count, turns) x window)
% 
% (c) Kevin Englehart, 1997
%   Revised by Xiaorong Zhang, 2014
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Feature_matrix] = tdfeats(DataSet,Nframe,WL,WInc)
% Parameters used in feature extraction The purpose of 
% these parameters are to normalize the feature values.
DEADZONE_ZC = 0.025;   % estimated baseline noise level for calculating zero crossings
DEADZONE_TURN = 0.015; % estimated noise level for calculating slope sign changes
SCALE_ZC = 15;      % parameter for zero crossing
SCALE_MAV = 2;      % parameter for mean absolute value
ruler = 1/WL;      % parameter for waveform length
rulersq = ruler^2;  % parameter for waveform length
lscale = WL/40;    % parameter for waveform length
tscale=(WL/40.0)*10;   % parameter for slope sign changes



Ntotal = size(DataSet,1);   %number of data points in each channel
Nsig = size(DataSet,2);     %number of channels
DataSet = DataSet - ones(Ntotal,1)*mean(DataSet);   %calculate mean of each channel, and subtract it from the raw EMG, so now the new mean of each channel should be zero

Feature_matrix=[];

for SigNum = 1:Nsig     % for each channel, calculate the four features
	clear mav dmav turns zero_count len;
	for frame = 1:Nframe
		zero_count(frame) = 0;
		len(frame) = 0;
		mav(frame) = 0;
		dmav(frame) = 0;
		turns(frame) = 0;
 
		index = 1 + (frame-1)*WInc;
		range = index:index+WL-1;
		mav(frame) = mean(abs(DataSet(range,SigNum)));
        
		flag1 = 1;
		flag2 = 1;
		for i = 1:WL-2;
			idx = index + i;
			fst = abs(DataSet(idx-1,SigNum));
			mid = abs(DataSet(idx,SigNum));
			lst = abs(DataSet(idx+1,SigNum));

% Compute Zero Crossings
			if ((DataSet(idx,SigNum)>=0 && DataSet(idx-1,SigNum)>=0) || (DataSet(idx,SigNum)<=0 && DataSet(idx-1,SigNum)<=0))
				flag1 = flag2;
			elseif ((mid<DEADZONE_ZC) && (fst<DEADZONE_ZC))
				flag1 = flag2;
			else
				flag1 = (-1)*flag2;
			end
			if (flag1~=flag2);
				zero_count(frame) = zero_count(frame) + 1;
				flag1 = flag2;
			end	
% Compute Turns (Slope Changes
			if((mid>fst && mid>lst) || (mid<fst && mid<lst))
% turns threshold
				if((abs(mid)-abs(fst))>DEADZONE_TURN || (abs(mid)-abs(lst))>DEADZONE_TURN)
					turns(frame)=turns(frame)+1;
				end
			end
% Compute Waveform Length
			len(frame) = len(frame) + sqrt(((fst-mid)/20.0)^2 + rulersq);
		end				
	end
   

% Scale the features to normalize for the neural network
	zero_count = (zero_count./SCALE_ZC)*40/WL;
% scaling based on 40 ms
	mav = mav/SCALE_MAV;
	len = (len-1)./lscale;
	turns = turns/tscale;

	Features(:,SigNum) = [mav(:)' len(:)' zero_count(:)' turns(:)']';
    Feature_matrix=[Feature_matrix;Features(:,SigNum)];
 
end

end

