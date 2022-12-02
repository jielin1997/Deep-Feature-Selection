function cwt2image_Ver2(data, image_save_path)
% ��һά���ݾ���С���任תΪ2άͼ��
% RGB����ͨ��ͨ�����Ƶõ�
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
    %ͼ�����ڱ߿�ռ䣬�� 'Border' �� 'tight' �� 'loose' ��ɡ�
    %��Ϊ 'loose' ʱ��ͼ�����ڰ���ͼ���е�ͼ����Χ�Ŀռ䡣
    %��Ϊ 'tight' ʱ��ͼ�����ڲ�����ͼ���е�ͼ����Χ���κοռ䡣
    %ͼ����ʾ�ĳ�ʼ�Ŵ��ʣ�ָ��Ϊ���ŷָ��Ķ��飬���а��� 'InitialMagnification' ��һ����ֵ������ 'fit'��
    %�����Ϊ 100���� imshow �� 100% �Ŵ�������ʾͼ��ÿ��ͼ�����ض�Ӧһ����Ļ���أ���
    %�����Ϊ 'fit'���� imshow ��������ͼ�����ʺϴ��ڡ�
    set (gcf,'Position',[0,0,image_size,image_size]); 
    %'Position'���趨plot���ͼƬ�ĳߴ硣��������Ϊ��xmin,ymin,width,height
    saveas(gcf,[image_save_path,'\',num2str(i_image),'.png']);
end

