function cwt2image(data, image_save_path)
% 将一维数据经过小波变换转为2维图像
% RGB三个通道通过复制得到
data_length = length(data);
wavename='cmor3-3'; 
wcf=centfrq(wavename); 
totalscal=128;
sample_num = 1200;
image_num = floor(data_length/sample_num);
cparam=2*wcf*totalscal;  
scal=cparam./(1:1:totalscal);
image_size = 224;
for i_image=1:1:image_num
    data_range = (1+sample_num*(i_image-1)):sample_num*i_image;
    coefs=cwt(data(data_range),scal,wavename);  
    coef=abs(coefs);
    image = coef./max(coef);
    image=imresize(image,[image_size,image_size]);
    rgbimage = zeros(image_size,image_size,3);
    rgbimage(:,:,1) = image;
    rgbimage(:,:,2) = image;
    rgbimage(:,:,3) = image;
%     rgbimage=uint8(rgbimage);
    imwrite(rgbimage,[image_save_path,'\',num2str(i_image),'.png']);
end

