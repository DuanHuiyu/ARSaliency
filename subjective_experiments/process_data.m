% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% data pre-processing file
% --------------------------------------------------------------------------

clc
clear all

%% 
des_path = '..\SARD\raw_subjective_data\gaze\';
if ~exist(des_path)
    mkdir(des_path)
end

raw_folder = fullfile('..\SARD\raw_subjective_data\raw_data\');
sub_folders = dir(fullfile(raw_folder,'*'));
for cnt = 1:size(sub_folders,1)
    cnt
    sub_folder_name = sub_folders(cnt).name;
    % subject name
    subject_name_split = strsplit(sub_folder_name,'_');
    subject_name = subject_name_split{1,1};
    
    if ~strcmp(sub_folder_name,'.') && ~strcmp(sub_folder_name,'..')
        % sub folder
        sub_folder = [raw_folder,sub_folder_name,'\eye_tracking\'];
        sub_files = dir(fullfile(sub_folder,'*.csv'));
        for cnt2 = 1:size(sub_files,1)
            sub_file_name = sub_files(cnt2).name
            img_name_split = strsplit(sub_file_name,{'__','.'});   % split '*.csv'
            img_name = img_name_split{1,end-1};
            % destination name
            des_name = [des_path,img_name,'_',subject_name,'.csv'];
            suffix = 0;
            while 1
                if ~isfile(des_name) 
                     % File does not exist.
                     copyfile([sub_folder,'\',sub_file_name],des_name)
                     break
                else
                     % File exists.
                     suffix = suffix+1;
                     des_name = [des_path,num2str(idx),'_',img_name,'_',subject_name,num2str(suffix),'.csv'];
                end
            end
        end
    end
end