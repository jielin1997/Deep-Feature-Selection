clc;
clear all;
close all;
 
drive_100 = load('normal_3_100.mat');
drive_108 = load('12k_Drive_End_IR007_3_108.mat');
drive_121 = load('12k_Drive_End_B007_3_121.mat');
drive_133 = load('12k_Drive_End_OR007@6_3_133.mat');
drive_172 = load('12k_Drive_End_IR014_3_172.mat');
drive_188 = load('12k_Drive_End_B014_3_188.mat');
drive_200 = load('12k_Drive_End_OR014@6_3_200.mat');
drive_212 = load('12k_Drive_End_IR021_3_212.mat');
drive_225 = load('12k_Drive_End_B021_3_225.mat');
drive_237 = load('12k_Drive_End_OR021@6_3_237.mat');

len_seg = 1024;
de_0 = drive_100.X100_DE_time(1:4:end);
N_end = floor(length(de_0)/len_seg)*len_seg;
de_1 = drive_108.X108_DE_time(1:N_end);
de_2 = drive_121.X121_DE_time(1:N_end);
de_3 = drive_133.X133_DE_time(1:N_end);
de_4 = drive_172.X172_DE_time(1:N_end);
de_5 = drive_188.X188_DE_time(1:N_end);
de_6 = drive_200.X200_DE_time(1:N_end);
de_7 = drive_212.X212_DE_time(1:N_end);
de_8 = drive_225.X225_DE_time(1:N_end);
de_9 = drive_237.X237_DE_time(1:N_end);

N_seg = N_end/len_seg;
X0 = zeros(len_seg,N_seg);
X1 = zeros(len_seg,N_seg);
for k = 1:N_seg
    idx = (k-1)*len_seg+(1:len_seg);
    X0(:,k) = de_0(idx);
    X1(:,k) = de_1(idx);
end

for k = 1:N_seg
    [wt,f,coi] = cwt(X0(:,k),'amor',12e3);
    wtn = abs(wt)/max(abs(wt(:)));
    wtnrs = imresize(wtn, [300, 300]);
    figure, imagesc(wtnrs);colormap jet;
    disp('hello');
end
