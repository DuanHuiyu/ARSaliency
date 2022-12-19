% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------

clc
clear all

columns = {'AR', 'BR', 'Superimposed'};

path_AR = '..\SARD\small_size\AR\';
path_BG = '..\SARD\small_size\BG\';
path_Superimposed = '..\SARD\small_size\Superimposed\';

path_write = 'train_test_split_\';
if ~exist(path_write)
    mkdir(path_write)
end

imgs = dir([path_Superimposed, '*.png']);

% train/test split 4:1
test = 4;   % 0,1,2,3,4

i = 0;
j = 0;
for cnt = 1:450
    temp = mod(cnt-1, 25);    % base: 25
    class = floor(temp / 5);   % sub class base: 5
    if class == test
        i = i+1;
        test_list(i) = cnt;
    else
        j = j+1;
        train_list(j) = cnt;
    end
end

AR_train = {};
BG_train = {};
Sperimposed_train = {};
AR_test = {};
BG_test = {};
Sperimposed_test = {};
for cnt = 1:length(imgs)
    img_name = imgs(cnt).name;
    temp_name = split(img_name, '.');   % the "split" function may be conflict with "baseline_saliency_models\tra_saliency_lib\Felzenszwalb_car&person\voc-release3.1\split.m" function.
                                        % if there is error here, please remove this path.
    temp_name2 = split(temp_name{1}, '_');
    BG_num = str2num(erase(temp_name2{1},'P'));
    test_or_not = find(test_list==BG_num);
    if isempty(test_or_not)
        AR_train = [AR_train, [temp_name2{3},'_',temp_name2{4},'.png']];
        BG_train = [BG_train, [temp_name2{1},'_',temp_name2{2},'.png']];
        Sperimposed_train = [Sperimposed_train, img_name];
    else
        AR_test = [AR_test, [temp_name2{3},'_',temp_name2{4},'.png']];
        BG_test = [BG_test, [temp_name2{1},'_',temp_name2{2},'.png']];
        Sperimposed_test = [Sperimposed_test, img_name];
    end
end

data = table(BG_train', AR_train', Sperimposed_train', 'VariableNames', columns);
writetable(data, [path_write, 'train',num2str(test),'.csv'])
data = table(BG_test', AR_test', Sperimposed_test', 'VariableNames', columns);
writetable(data, [path_write, 'test',num2str(test),'.csv'])