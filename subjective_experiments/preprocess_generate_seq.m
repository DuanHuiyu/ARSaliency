% --------------------------------------------------------------------------
% Saliency in Augmented Reality
% Huiyu Duan, Wei Shen, Xiongkuo Min, Danyang Tu, Jing Li, and Guangtao Zhai
% ACM International Conference on Multimedia (ACM MM 2022)
% --------------------------------------------------------------------------
% code to control the sequence in Unity
% --------------------------------------------------------------------------

% AR: 1-450  -1
% BG: 1-450  -1
% mixing: 1-3  -1
% mixing_value in Unity: mixing_value = { 0.25f, 0.50f, 0.75f };

clc
clear all

img_num = 450;

% load experiment_seq.mat

%% saliency

cnt = 0;
seq = [];

for seq_ar = 1:img_num
    seq_bg = seq_ar;
    seq_mixing = [0,1,2];
    temp_random_index = randperm(size(seq_mixing,2));
    seq_mixing = seq_mixing(:,temp_random_index);
    cnt = cnt+1;
    seq1(:,cnt) = [seq_mixing(1),seq_ar-1,seq_bg-1];
    seq2(:,cnt) = [seq_mixing(2),seq_ar-1,seq_bg-1];
    seq3(:,cnt) = [seq_mixing(3),seq_ar-1,seq_bg-1];
end


%% seq1
text1_1 = [];
text1_2 = [];
text1_3 = [];
for cnt = 1:size(seq1,2)
    text1_1 = [text1_1,num2str(seq1(1,cnt)),','];    % mixing seq
    text1_2 = [text1_2,num2str(seq1(2,cnt)),','];    % ar seq
    text1_3 = [text1_3,num2str(seq1(3,cnt)),','];    % bg seq
end

% random seq
random_index1 = randperm(size(seq1,2));
seq1 = seq1(:,random_index1);

% text to unity
text1_saliency_1 = [];
text1_saliency_2 = [];
text1_saliency_3 = [];
for cnt = 1:size(seq1,2)
    text1_saliency_1 = [text1_saliency_1,num2str(seq1(1,cnt)),','];    % mixing seq
    text1_saliency_2 = [text1_saliency_2,num2str(seq1(2,cnt)),','];    % ar seq
    text1_saliency_3 = [text1_saliency_3,num2str(seq1(3,cnt)),','];    % bg seq
end
% text to unity (split)
text1_saliency_1_part1 = [];
text1_saliency_2_part1 = [];
text1_saliency_3_part1 = [];
text1_saliency_1_part2 = [];
text1_saliency_2_part2 = [];
text1_saliency_3_part2 = [];
text1_saliency_1_part3 = [];
text1_saliency_2_part3 = [];
text1_saliency_3_part3 = [];
for cnt = 1:size(seq1,2)/3
    text1_saliency_1_part1 = [text1_saliency_1_part1,num2str(seq1(1,cnt)),','];    % mixing seq
    text1_saliency_2_part1 = [text1_saliency_2_part1,num2str(seq1(2,cnt)),','];    % ar seq
    text1_saliency_3_part1 = [text1_saliency_3_part1,num2str(seq1(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq1,2)/3+1):(size(seq1,2)/3*2)
    text1_saliency_1_part2 = [text1_saliency_1_part2,num2str(seq1(1,cnt)),','];    % mixing seq
    text1_saliency_2_part2 = [text1_saliency_2_part2,num2str(seq1(2,cnt)),','];    % ar seq
    text1_saliency_3_part2 = [text1_saliency_3_part2,num2str(seq1(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq1,2)/3*2+1):size(seq1,2)
    text1_saliency_1_part3 = [text1_saliency_1_part3,num2str(seq1(1,cnt)),','];    % mixing seq
    text1_saliency_2_part3 = [text1_saliency_2_part3,num2str(seq1(2,cnt)),','];    % ar seq
    text1_saliency_3_part3 = [text1_saliency_3_part3,num2str(seq1(3,cnt)),','];    % bg seq
end
cnt

%% seq2
text2_1 = [];
text2_2 = [];
text2_3 = [];
for cnt = 1:size(seq2,2)
    text2_1 = [text2_1,num2str(seq2(1,cnt)),','];    % mixing seq
    text2_2 = [text2_2,num2str(seq2(2,cnt)),','];    % ar seq
    text2_3 = [text2_3,num2str(seq2(3,cnt)),','];    % bg seq
end

% random seq
random_index2 = randperm(size(seq2,2));
seq2 = seq2(:,random_index2);

% text to unity
text2_saliency_1 = [];
text2_saliency_2 = [];
text2_saliency_3 = [];
for cnt = 1:size(seq2,2)
    text2_saliency_1 = [text2_saliency_1,num2str(seq2(1,cnt)),','];    % mixing seq
    text2_saliency_2 = [text2_saliency_2,num2str(seq2(2,cnt)),','];    % ar seq
    text2_saliency_3 = [text2_saliency_3,num2str(seq2(3,cnt)),','];    % bg seq
end
% text to unity (split)
text2_saliency_1_part1 = [];
text2_saliency_2_part1 = [];
text2_saliency_3_part1 = [];
text2_saliency_1_part2 = [];
text2_saliency_2_part2 = [];
text2_saliency_3_part2 = [];
text2_saliency_1_part3 = [];
text2_saliency_2_part3 = [];
text2_saliency_3_part3 = [];
for cnt = 1:size(seq2,2)/3
    text2_saliency_1_part1 = [text2_saliency_1_part1,num2str(seq2(1,cnt)),','];    % mixing seq
    text2_saliency_2_part1 = [text2_saliency_2_part1,num2str(seq2(2,cnt)),','];    % ar seq
    text2_saliency_3_part1 = [text2_saliency_3_part1,num2str(seq2(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq2,2)/3+1):(size(seq2,2)/3*2)
    text2_saliency_1_part2 = [text2_saliency_1_part2,num2str(seq2(1,cnt)),','];    % mixing seq
    text2_saliency_2_part2 = [text2_saliency_2_part2,num2str(seq2(2,cnt)),','];    % ar seq
    text2_saliency_3_part2 = [text2_saliency_3_part2,num2str(seq2(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq2,2)/3*2+1):size(seq2,2)
    text2_saliency_1_part3 = [text2_saliency_1_part3,num2str(seq2(1,cnt)),','];    % mixing seq
    text2_saliency_2_part3 = [text2_saliency_2_part3,num2str(seq2(2,cnt)),','];    % ar seq
    text2_saliency_3_part3 = [text2_saliency_3_part3,num2str(seq2(3,cnt)),','];    % bg seq
end
cnt


%% seq3
text3_1 = [];
text3_2 = [];
text3_3 = [];
for cnt = 1:size(seq3,2)
    text3_1 = [text3_1,num2str(seq3(1,cnt)),','];    % mixing seq
    text3_2 = [text3_2,num2str(seq3(2,cnt)),','];    % ar seq
    text3_3 = [text3_3,num2str(seq3(3,cnt)),','];    % bg seq
end

% random seq
random_index3 = randperm(size(seq3,2));
seq3 = seq3(:,random_index3);

% text to unity
text3_saliency_1 = [];
text3_saliency_2 = [];
text3_saliency_3 = [];
for cnt = 1:size(seq3,2)
    text3_saliency_1 = [text3_saliency_1,num2str(seq3(1,cnt)),','];    % mixing seq
    text3_saliency_2 = [text3_saliency_2,num2str(seq3(2,cnt)),','];    % ar seq
    text3_saliency_3 = [text3_saliency_3,num2str(seq3(3,cnt)),','];    % bg seq
end
% text to unity (split)
text3_saliency_1_part1 = [];
text3_saliency_2_part1 = [];
text3_saliency_3_part1 = [];
text3_saliency_1_part2 = [];
text3_saliency_2_part2 = [];
text3_saliency_3_part2 = [];
text3_saliency_1_part3 = [];
text3_saliency_2_part3 = [];
text3_saliency_3_part3 = [];
for cnt = 1:size(seq3,2)/3
    text3_saliency_1_part1 = [text3_saliency_1_part1,num2str(seq3(1,cnt)),','];    % mixing seq
    text3_saliency_2_part1 = [text3_saliency_2_part1,num2str(seq3(2,cnt)),','];    % ar seq
    text3_saliency_3_part1 = [text3_saliency_3_part1,num2str(seq3(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq3,2)/3+1):(size(seq3,2)/3*2)
    text3_saliency_1_part2 = [text3_saliency_1_part2,num2str(seq3(1,cnt)),','];    % mixing seq
    text3_saliency_2_part2 = [text3_saliency_2_part2,num2str(seq3(2,cnt)),','];    % ar seq
    text3_saliency_3_part2 = [text3_saliency_3_part2,num2str(seq3(3,cnt)),','];    % bg seq
end
cnt
for cnt = (size(seq3,2)/3*2+1):size(seq3,2)
    text3_saliency_1_part3 = [text3_saliency_1_part3,num2str(seq3(1,cnt)),','];    % mixing seq
    text3_saliency_2_part3 = [text3_saliency_2_part3,num2str(seq3(2,cnt)),','];    % ar seq
    text3_saliency_3_part3 = [text3_saliency_3_part3,num2str(seq3(3,cnt)),','];    % bg seq
end
cnt