function cwt2image_Ver2(data, image_save_path)
% 将一维数据经过小波变换转为2维图像
% RGB三个通道通过复制得到
data_length = length(data);
wavename='amor'; 
len_seg = 1024;
image_num = floor(data_length/len_seg);
image_size = 300;
for i_image=1:1:image_num
    data_range = len_seg*(i_image-1)+(1:len_seg);
    [wt,f,coi] = cwt(data(data_range),wavename,len_seg);
    wtn = abs(wt)/max(abs(wt(:)));
    wtnrs = imresize(wtn, [image_size, image_size]);
%     figure;
%     h = heatmap(wtnrs);
%     h.GridVisible = 'off';
%     h.Colormap = jet;
%     YourYticklabel=cell(size(h.YDisplayLabels));
%     [YourYticklabel{:}]=deal('');
%     h.YDisplayLabels=YourYticklabel;
%     h.XDisplayLabels=YourYticklabel;
%     clear img
%     saveas(h,[image_save_path,'\',num2str(i_image),'.png']);
%     figure;
    imshow(wtnrs,'border','tight','initialmagnification','fit');colormap jet;
    %图窗窗口边框空间，由 'Border' 和 'tight' 或 'loose' 组成。
    %设为 'loose' 时，图窗窗口包含图窗中的图像周围的空间。
    %设为 'tight' 时，图窗窗口不包含图窗中的图像周围的任何空间。
    %图像显示的初始放大倍率，指定为逗号分隔的对组，其中包含 'InitialMagnification' 和一个数值标量或 'fit'。
    %如果设为 100，则 imshow 在 100% 放大倍率下显示图像（每个图像像素对应一个屏幕像素）。
    %如果设为 'fit'，则 imshow 缩放整个图像以适合窗口。
    set (gcf,'Position',[0,0,image_size,image_size]); 
    %'Position'：设定plot输出图片的尺寸。参数含义为：xmin,ymin,width,height
    saveas(gcf,[image_save_path,'\',num2str(i_image),'.png']);
end

