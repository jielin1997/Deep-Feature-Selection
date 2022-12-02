%% ÿ��ͼһ��figure��ʾ
clear,clc
%EfficientNet��ÿ��stage�ѵ����������
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% ѡ����camͼ��Ӧ������
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
% subplot(4,2,1);
imshow('cam_images\bottle\broken_large\000.png\000.png');%��ʾԭͼ
for i=1:length(block_stage)-1
    s = block_stage(i);%start����i��stage��s��ʼ
    e = block_stage(i+1) - 1;%end����i��stage��e����
    img = zeros(900,900);%һ���յĻ���
    %����level���ݣ�������img������
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp = block',num2str(j),';']);
        img = img + temp;
        % ��������level����
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%ƽ��
%     subplot(4,2,i+1);
    figure;
    h = heatmap(img, 'ColorbarVisible','off');%��ͼ
    h.Colormap = jet;
%     h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end
%% ����subplot��ʾ
clear,clc
%EfficientNet��ÿ��stage�ѵ����������
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% ѡ����camͼ��Ӧ������
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
subplot(4,2,1);
%��ʾԭͼ��Ҫ�Լ�ȥ���ݼ��аѶ�Ӧ��ͼƬ���Ƶ����Ŀ¼��
imshow('cam_images\bottle\broken_large\000.png\000.png');
for i=1:length(block_stage)-1
    s = block_stage(i);%start����i��stage��s��ʼ
    e = block_stage(i+1) - 1;%end����i��stage��e����
    img = zeros(900,900);%һ���յĻ���
    %����level���ݣ�������img������
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp =block',num2str(j),';']);
        img = img + temp;
        % ��������level����
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%ƽ��
    subplot(4,2,i+1);
%     figure;
    h = heatmap(img);%��ͼ
    h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    h.Colormap = jet;
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end

%% ����subplot��ʾ2
clear,clc
%EfficientNet��ÿ��stage�ѵ����������
block_stage_b4 = [0, 2, 4, 4, 6, 6, 8, 2]; 
% ѡ����camͼ��Ӧ������
block_stage = cumsum(block_stage_b4);
mat_path = 'cam_images\bottle\broken_large\000.png\blocks';
figure;
subplot(2,4,1);
%��ʾԭͼ��Ҫ�Լ�ȥ���ݼ��аѶ�Ӧ��ͼƬ���Ƶ����Ŀ¼��
imshow('cam_images\bottle\broken_large\000.png\000.png');
for i=1:length(block_stage)-1
    s = block_stage(i);%start����i��stage��s��ʼ
    e = block_stage(i+1) - 1;%end����i��stage��e����
    img = zeros(900,900);%һ���յĻ���
    %����level���ݣ�������img������
    for j = s:e
        name_path = strcat(mat_path,'\block',num2str(j),'.mat');
        load(name_path);
        eval(['block',num2str(j),'= squeeze(grayscale_cam(1,:,:));']);
        eval(['temp =block',num2str(j),';']);
        img = img + temp;
        % ��������level����
        eval(['clear block',num2str(j)]);
        clear temp;
    end
    img = img / block_stage_b4(i+1);%ƽ��
    subplot(2,4,i+1);
%     figure;
    h = heatmap(img);%��ͼ
    h.Title = ['stage',num2str(i+1)];
    h.GridVisible = 'off';
    h.Colormap = jet;
    YourYticklabel=cell(size(h.YDisplayLabels));
    [YourYticklabel{:}]=deal('');
    h.YDisplayLabels=YourYticklabel;
    h.XDisplayLabels=YourYticklabel;
    clear img
end
