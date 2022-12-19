function [fixation_position,fixation_times,fixation_duration] = calPanoGaze2Fixation(latVec,longVec)
% ------------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% ------------------------------------------------------------------------------
% follow the method in: Active vision in immersive, 360° real‑world environments
% used to calculate fixaiton from gaze point;
% fixation_position: (latitude vection, longtitude vector)
% ------------------------------------------------------------------------------

% --- default parameter ---
% slidingWindow = 9;  % (inside ~80ms) floor(0.08*samplingFreq)
slidingWindow = 7;  % (inside ~80ms) floor(0.08*samplingFreq)
fixedThreshVal = 50;
fixedThreshDuration = 0.100;
fixedSpatialInterval = 1;
fixedDurationInterval = 150;

% samplingFreq = 120;
samplingFreq = 90;
% -------------------------

X = longVec;    % 1*N
Y = latVec;    % 1*N


% Lowpass filter window length
% smoothInt = minSaccadeDur; % in seconds

% Span of filter
% span = ceil(smoothInt*samplingFreq);
span = slidingWindow;

% Calculate unfiltered data
Xorg = X;
Yorg = Y;

velXorg = [0 diff(X)]/samplingFreq; %removed angleInPixelsV
velYorg = [0 diff(Y)]/samplingFreq; %remvoed angleInPixelsV
velOrg = sqrt(velXorg.^2 + velYorg.^2);

% Pixel values, velocities, and accelerations
N = 2;                 % Order of polynomial fit
F = 2*ceil(span)-1;    % Window length
[b,g] = sgolay(N,F);   % Calculate S-G coefficients
Nf = F;

% Calculate the velocity and acceleration
tempX = conv(X, g(:,1)', 'same');
tempX(1:(Nf-1)/2) = X(1:(Nf-1)/2);
tempX(end-(Nf-3)/2:end) = X(end-(Nf-3)/2:end);
tempY = conv(Y, g(:,1)', 'same');
tempY(1:(Nf-1)/2) = Y(1:(Nf-1)/2);
tempY(end-(Nf-3)/2:end) = Y(end-(Nf-3)/2:end);
X = tempX;
Y = tempY;

tempX = conv(X, -g(:,2)', 'same');
tempX(1:(Nf-1)/2) = 0;
tempX(end-(Nf-3)/2:end) = 0;
tempY = conv(Y, -g(:,2)', 'same');
tempY(1:(Nf-1)/2) = 0;
tempY(end-(Nf-3)/2:end) = 0;
velX = tempX;
velY = tempY;
vel = sqrt(velX.^2 + velY.^2)*samplingFreq;

% define potential fixation with: vel<fixedThreshVal
potential_fixation_idx = find(vel<fixedThreshVal);
potential_fixation = vel(potential_fixation_idx);

% concatenate fixations
len = length(vel);
fixation_idx = zeros(1,len);
fixation_idx(potential_fixation_idx) = 1;
fixation_idx(find(Xorg==0&Yorg==0)) = 0;    % exclude (0,0)

[fixation_times] = BehavioralIndex(find(fixation_idx));

[fixation_position,fixation_times,fixation_position_concatenate,fixation_times_concatenate] = calFixCenter(Xorg,Yorg,fixation_times,fixedDurationInterval,samplingFreq);

fix_dur = diff(fixation_times,1)+1;
too_short = find(fix_dur < fixedThreshDuration*samplingFreq);
fixation_times(:,too_short) = [];
fixation_duration = diff(fixation_times,1)+1;
fixation_position(:,too_short) = [];

fix_dur2 = diff(fixation_times_concatenate,1)+1;
too_short = find(fix_dur2 < fixedThreshDuration*samplingFreq);
fixation_times_concatenate(:,too_short) = [];
fixation_position_concatenate(:,too_short) = [];

end

function [fixation_position,fixation_times,fixation_position_concatenate,fixation_times_concatenate] = calFixCenter(longVec,latVec,fixation_times,fixedDurationInterval,samplingFreq)
fixation_times = fixation_times;
for idx = 1:size(fixation_times,2)
    fixation(1,idx) = mean(latVec(fixation_times(1,idx):fixation_times(2,idx)));
    fixation(2,idx) = mean(longVec(fixation_times(1,idx):fixation_times(2,idx)));
end
cnt = 1;
for idx = 1:size(fixation_times,2)
    
    if idx == 1
        fixation_position_concatenate(:,1) = fixation(:,1);
        fixation_times_concatenate(:,1) = fixation_times(:,1);
        cnt = cnt+1;
        continue;
    end
    angle = SphereDist([fixation(2,idx),fixation(1,idx)-90],[fixation(2,idx-1),fixation(1,idx-1)-90]);
    if (angle<1) && (fixation_times(1,idx)-fixation_times(2,idx-1))<(fixedDurationInterval*samplingFreq/1000)    % less than 1 degree and 150ms, then concatenate two fixations
        cnt = cnt-1;
        fixation_position_concatenate(:,cnt) = [mean([fixation_position_concatenate(1,cnt),fixation(1,idx)],'all');mean([fixation_position_concatenate(2,cnt),fixation(2,idx)],'all')];
        fixation_times_concatenate(2,cnt) = fixation_times(2,idx);
    else
        fixation_position_concatenate(:,cnt) = fixation(:,idx);
        fixation_times_concatenate(:,cnt) = fixation_times(:,idx);
    end
    cnt = cnt+1;
end
fixation_position = fixation;
end


% calculate sphere angle based on sphere cosine
% x: point(Longtitude,Latitude), y: point(Longtitude,Latitude)
function angle = SphereDist(x,y)
x = D2R(x);
y = D2R(y);
DeltaS = acos(cos(x(2))*cos(y(2))*cos(x(1)-y(1))+sin(x(2))*sin(y(2)));
angle = R2D(DeltaS);
function h = HaverSin(theta)
    h=sin(theta/2)^2;
end
function rad = D2R(theta)
    rad = theta*pi/180;
end
function theta = R2D(rad)
    theta = rad*180/pi;
end
end
% calculate sphere angle based on Haversine
% x: (Longtitude,Latitude), y: (Longtitude,Latitude)
function angle = SphereDist2(x,y)
if nargin < 3
    R = 6378.137;
end
x = D2R(x);
y = D2R(y);
h = HaverSin(abs(x(2)-y(2)))+cos(x(2))*cos(y(2))*HaverSin(abs(x(1)-y(1)));
angle = 2 * asin(sqrt(h));
angle = R2D(angle);
function h = HaverSin(theta)
    h=sin(theta/2)^2;
end
function rad = D2R(theta)
    rad = theta*pi/180;
end
function theta = R2D(rad)
    theta = rad*180/pi;
end
end


% detect fixation time
function [behaviortime] = BehavioralIndex(behavind)
[behaveind]=findgaps(behavind);
if isempty(behavind)
    behaviortime = [];
    return
end
for index=1:size(behaveind,1)
    rowfixind = behaveind(index,:);
    rowfixind(rowfixind == 0) = [];
    behaviortime(:,index) = [rowfixind(1);rowfixind(end)];
end
end

function [broken_ind]=findgaps(input_ind)
% finds gaps (greater than 0) in between indeces in a vector
%
% rechecked for bugs by SDK on 1/5/2017

gaps =find(abs(diff(input_ind)) > 1);
broken_ind = zeros(length(gaps),50);
if ~isempty(gaps)
    for gapind = 1:length(gaps)+1;
        if gapind == 1;
            temp = input_ind(1:gaps(gapind));
        elseif gapind == length(gaps)+1
            temp = input_ind(gaps(gapind-1)+1:end);
        else
            temp = input_ind(gaps(gapind-1)+1:gaps(gapind));
        end
        broken_ind(gapind,1:length(temp)) = temp;
    end
else
    broken_ind = input_ind;
end

end