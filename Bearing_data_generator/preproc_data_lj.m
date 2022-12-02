%% �������Լ�
clear,clc;
% bearing_loads = ['0','1','2','3'];
bearing_loads = '3';
set = 'test';
fault_positions = ["B", "IR", "OR"];
fault_diameters = ["007","014","021"];
original_path = 'original_data';
image_path = 'images';
type = '12k_Drive_End_';
% for i_load=1
for i_load=1:length(bearing_loads)
    %����load�ֱ𴴽��ļ���
    load_path = [image_path,'\cond_',bearing_loads(i_load)];
    if ~isfolder(load_path)
        mkdir(load_path);
    else
%         rmdir(load_path,'s');
%         mkdir(load_path);
        disp(['�Ѵ�������',num2str(i_load),'�ļ���']);
    end
    %�������Լ�orѵ�����ļ���
    set_path = [load_path,'\',char(set)];
    if ~isfolder(set_path)
        mkdir(set_path);
    else
        rmdir(set_path,'s');
        mkdir(set_path);
    end
    %��λ������λ��
    for i_position=1:length(fault_positions)
        %��λ�����ϴ�С
        for i_diameter = 1:length(fault_diameters)
            %��ȡ��Ӧ���ļ�
            if fault_positions(i_position)=="OR" %�������λ��ΪOR����Ҫ��@
                file_name = [type,char(fault_positions(i_position)),char(fault_diameters(i_diameter)),'@6','_',bearing_loads(i_load)];
            else
                file_name = [type,char(fault_positions(i_position)),char(fault_diameters(i_diameter)),'_',bearing_loads(i_load)];
            end
            file_path = [char(original_path),'\',file_name];
            matObj = matfile(file_path);
            %ÿ��mat�ļ��а���������ݣ�ȡ����FE��β������(���ȶ�)
            data_info = whos('-file',file_path);
            data_name = {data_info.name};
            for i_data_name = 1:length(data_name)
                %�ҵ�DE������
                if contains(char(data_name(i_data_name)),'DE')
                    bearing_name = char(data_name(i_data_name));
                end
            end
            eval(['data = matObj.',bearing_name,';'])
            image_save_path = [set_path,'\',file_name];
            if ~isfolder(image_save_path)
                mkdir(image_save_path);
            else
                rmdir(image_save_path,'s');
                mkdir(image_save_path);
            end
            cwt2image_Ver2(data,image_save_path);
        end
    end      
end
%% ����ѵ����
clear,clc;
% bearing_loads = ['0','1','2','3'];
bearing_loads = '3';
set = 'train';
original_path = 'original_data';
image_path = 'images';
% for i_load=1
for i_load=1:length(bearing_loads)
    %����load�ֱ𴴽��ļ���
    load_path = [image_path,'\cond_',bearing_loads(i_load)];
    if ~isfolder(load_path)
        mkdir(load_path);
    else
        disp(['�Ѵ�������',num2str(i_load),'�ļ���']);
    end
    %�������Լ�orѵ�����ļ���
    set_path = [load_path,'\',char(set)];
    if ~isfolder(set_path)
        mkdir(set_path);
    else
        mkdir(set_path);
    end
    good_path = [set_path,'\good'];
    file_name = ['normal','_',bearing_loads(i_load)];
    file_path = [original_path,'\',file_name];
    matObj = matfile(file_path);
    %ÿ��mat�ļ��а���������ݣ�ȡ����FE��β������(���ȶ�)
    data_info = whos('-file',file_path);
    data_name = {data_info.name};
    for i_data_name = 1:length(data_name)
        if contains(char(data_name(i_data_name)),'DE')
            bearing_name = char(data_name(i_data_name));
        end
    end
    eval(['data = matObj.',bearing_name,';'])
    image_save_path = good_path;
    if ~isfolder(image_save_path)
        mkdir(image_save_path);
    else
        rmdir(image_save_path,'s');
        mkdir(image_save_path);
    end
    %normal������
    step = 4;
    data = data(1:step:end);
    cwt2image_Ver2(data,image_save_path);
end





