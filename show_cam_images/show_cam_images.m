%% 每张图一个figure显示
clear,clc
%EfficientNet中每个stage堆叠的网络层数
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% 选择与cam图对应的网络
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
% subplot(4,2,1);
imshow('cam_images\bottle\broken_large\000.png\000.png');%显示原图
for i=1:length(block_stage)-1
    s = block_stage(i);%start，第i个stage从s开始
    e = block_stage(i+1) - 1;%end，第i个stage在e结束
    img = zeros(900,900);%一个空的画板
    %导入level数据，并存在img画板内
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp = block',num2str(j),';']);
        img = img + temp;
        % 清除多余的level变量
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%平均
%     subplot(4,2,i+1);
    figure;
    h = heatmap(img, 'ColorbarVisible','off');%热图
    h.Colormap = jet;
%     h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end
%% 按照subplot显示
clear,clc
%EfficientNet中每个stage堆叠的网络层数
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% 选择与cam图对应的网络
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
subplot(4,2,1);
%显示原图，要自己去数据集中把对应的图片复制到这个目录下
imshow('cam_images\bottle\broken_large\000.png\000.png');
for i=1:length(block_stage)-1
    s = block_stage(i);%start，第i个stage从s开始
    e = block_stage(i+1) - 1;%end，第i个stage在e结束
    img = zeros(900,900);%一个空的画板
    %导入level数据，并存在img画板内
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp =block',num2str(j),';']);
        img = img + temp;
        % 清除多余的level变量
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%平均
    subplot(4,2,i+1);
%     figure;
    h = heatmap(img);%热图
    h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    h.Colormap = jet;
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end

%% 按照subplot显示2
clear,clc
%EfficientNet中每个stage堆叠的网络层数
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% 选择与cam图对应的网络
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
subplot(2,4,1);
%显示原图，要自己去数据集中把对应的图片复制到这个目录下
imshow('cam_images\bottle\broken_large\000.png\000.png');
for i=1:length(block_stage)-1
    s = block_stage(i);%start，第i个stage从s开始
    e = block_stage(i+1) - 1;%end，第i个stage在e结束
    img = zeros(900,900);%一个空的画板
    %导入level数据，并存在img画板内
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp =block',num2str(j),';']);
        img = img + temp;
        % 清除多余的level变量
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%平均
    subplot(2,4,i+1);
%     figure;
    h = heatmap(img);%热图
    h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    h.Colormap = jet;
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end
