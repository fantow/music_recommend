%% FINISH
%% PART Ⅰ
clear all
clc
warning('off') %关闭函数警告
%% 读取歌曲名称
dirs=dir('D:\音乐推荐\music'); % 用你需要的目录以及文件扩展名替换。读取某个目录的指定类型文件列表，返回结构数组。
dircell=struct2cell(dirs)' ; % 结构体(struct)转换成元胞类型(cell)，转置一下是让文件名按列排列。
song_filenames=dircell(3:end,1) % 第一列是文件名
song_filename_num=size(song_filenames,1);
%% 歌曲随机排序
For_sort_num=(randperm(song_filename_num))';
song_filenames(:,2)=num2cell(For_sort_num);
For_song_filenames=sortrows(song_filenames,2);
song_filenames=For_song_filenames(:,1);


save('all_song_list.mat','song_filenames');   
%% 开始循环
for ii=1:22
    filedir=[];
    load all_song_list.mat;
    filename=song_filenames{ii};
    fle=[filedir filename];
    [x,fs]=wavread(fle);
    For_sound=x;
    filename_numbers=size(filename,2); 
    filename=filename(1:filename_numbers-4);
%% 计算特征值   
    a=size(x,1);
    rand_Min=a/6;
    rand_Max=a/2;
    rand_a=round(rand(1)*(rand_Max-rand_Min)+rand_Min);
    b=x(rand_a:rand_a+18000,:);
    x=b;
%function  mfc=mfcc(x)
 %%%%%%%%%%%%%%%%%%%%%%%%%
 %对输入的语音序列进行mfcc参数提取，返回mfcc参数和一阶差分mfcc参数，mel滤波器的阶数为24
 %fft变换长度为256，采样频率为8000HZ,对x 256点分为一帧
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    bank=melbankm(24,256,8000,0,0.5,'m');
 %归一化mel滤波器组参数
    bank=full(bank);
    bank=bank/max(bank(:));
 %DCT系数，12*24
    for k=1:12
         n=0:23;
        dctcoef(k,:)=cos((2*n+1)*k*pi/(2*24));
    end
 %归一化倒谱提升窗口
    w=1+6*sin(pi*[1:12]./12);
    w=w/max(w);
 %预加重滤波器
    xx=double(x);
    xx=filter([1 -0.9375],1,xx);
 %语音信号分帧
        xx=enframe(xx,256,80);
 %计算每帧的mfcc参数
    for i=1:size(xx,1) %
         y=xx(i,:);
         s=y'.*hamming(256);
         t=abs(fft(s));
         t=t.^2;%计算能量
         c1=dctcoef*log(bank*t(1:129));%dctcoef为dct系数，bank归一化mel滤波器组系数
         c2=c1.*w';%w为归一化倒谱提升窗口
         m(i,:)=c2';
    end
 %差分系数
    dtm=zeros(size(m));
    for i=3:size(m,1)-2;
        dtm(i,:)=-2*m(i-2,:)-m(i-1,:)+m(i+1,:)+2*m(i+2,:);
    end
     dtm=dtm/3;
 %合并mfcc参数和一阶差分mfcc参数 
    mfc=[m dtm];
 %去除首尾两帧，因为这两帧的一阶差分参数为0
    mfc=mfc(3:size(m,1)-2,:);
 %% 播放音乐并显示歌名
  if(ii<9)
       sound(For_sound,fs);
       fprintf('==================================================\n');
       fprintf('正在播放第%d首歌曲:%s\n',ii,filename);
       classify=input('Do you like this song?  Y/N:','s');
       switch (classify)
     case 'Y'
           T=[ones(size(mfc,1),1),mfc];
           a=1;
     case 'N'
            T=[zeros(size(mfc,1),1),mfc];
           a=0;
       end
  end
 %T5=[ones(size(mfc,1),1),mfc];
 %T=[zeros(size(mfc,1),1),mfc];
   %eval(['T',num2str(i),'=','T',';']);
  % clear T;
 %% 保存特征值 将没被划分类的默认为1
 if(ii>=9)
     T=[ones(size(mfc,1),1),mfc];
 end
    fprintf('正在保存第%d首歌曲的特征值。歌曲名称:%s\n',ii,filename);
    S2='.mat';
    save_name=strcat(filename,S2);
    save(save_name,'T');
 %%  停止歌曲
clear all
clc
end











%% PART Ⅱ
%% 加入BP神经网络
load all_song_list.mat
%% 导入.mat文件,所有文件中的T // 提取文件名，并去掉文件后缀 //合并特征信号
load_T1=song_filenames{1};
num_1=size(load_T1,2);
load_T1=load_T1(1:num_1-4);  %1
load(load_T1);
data(1:443,:)=T(1:443,:); %A
clearvars -EXCEPT data song_filenames;

load_T2=song_filenames{2};
num_2=size(load_T2,2);
load_T2=load_T2(1:num_2-4);%2
load(load_T2);
data(444:886,:)=T(1:443,:); %B
clearvars -EXCEPT data song_filenames;

load_T3=song_filenames{3};
num_3=size(load_T3,2);
load_T3=load_T3(1:num_3-4);%3
load(load_T3);
data(887:1329,:)=T(1:443,:); %C
clearvars -EXCEPT data song_filenames;

load_T4=song_filenames{4};
num_4=size(load_T4,2);
load_T4=load_T4(1:num_4-4);%4
load(load_T4);
data(1330:1772,:)=T(1:443,:); %D
clearvars -EXCEPT data song_filenames;

load_T5=song_filenames{5};
num_5=size(load_T5,2);
load_T5=load_T5(1:num_5-4);%5
load(load_T5);
data(1773:2215,:)=T(1:443,:); %E
clearvars -EXCEPT data song_filenames;

load_T6=song_filenames{6};
num_6=size(load_T6,2);
load_T6=load_T6(1:num_6-4);%6
load(load_T6);
data(2216:2658,:)=T(1:443,:); %F
clearvars -EXCEPT data song_filenames;

load_T7=song_filenames{7};
num_7=size(load_T7,2);
load_T7=load_T7(1:num_7-4);%7
load(load_T7);
data(2659:3101,:)=T(1:443,:); %G
clearvars -EXCEPT data song_filenames;

load_T8=song_filenames{8};
num_8=size(load_T8,2);
load_T8=load_T8(1:num_8-4);%8
load(load_T8); %load T
data(3102:3544,:)=T(1:443,:); %H
clearvars -EXCEPT data song_filenames;
%% 
   % data(1:443,:)=A(1:443,:); %A
   %data(444:886,:)=B(1:443,:); %B
   %data(887:1329,:)=C(1:443,:); %C
   %data(1330:1772,:)=D(1:443,:); %D
   % data(1773:2215,:)=E(1:443,:); %E
   % data(2216:2658,:)=F(1:443,:); %F
   % data(2659:3101,:)=G(1:443,:); %G
   % data(3102:3544,:)=H(1:443,:); %H

%% 输入输出数据
input=data(:,2:25);
output=data(:,1);
%% 随机选择2835组训练数据，709组测试数据
k=rand(1,3544);
[m,n]=sort(k);
input_train=input(n(1:2835),:)';
output_train=output(n(1:2835),:)';
input_test=input(n(2836:3544),:)';
output_test=output(n(2836:3544),:)';

%% 训练数据归一化
[inputn,inputs]=mapminmax(input_train);
[outputn,outputs]=mapminmax(output_train); %(E)

%% 神经网络搭建

net=newff(inputn,outputn,[30,25]); %(E)

%% 设置参数
net.trainParam.epochs = 50;
net.trainParam.goal = 1e-4;
net.trainParam.lr = 0.0001;
net.trainParam.max_fail = 2;
%% 网络训练

fprintf('正在训练网络......:\n');
net=train(net,inputn,outputn);

%% 网络预测
inputn_test=mapminmax('apply',input_test,inputs);

result=sim(net,inputn_test);

Result=mapminmax('reverse',result,outputs);
%% 误差绘图
sum=0;
Er=0;
for i=1:size(output_test,2)
    if Result(i)>=0.65
        Result(i)=1;
    end
    if Result(i)<0.65
        Result(i)=0;
    end
end
error=Result-output_test;
for i=1:size(output_test,2)
   if error(i)~=0
    sum=sum+1;
   end
end
Er=1-(sum/size(output_test,2));
figure(1);
stem(error,'-.dg','fill');
%plot(error,'-*');
string = {'正确率';['Er='  num2str(Er)]};
title(string);
ylabel('误差','fontsize',12);
xlabel('样本','fontsize',12);
%% 将all_song_filenames 全部标为-1
%feature=[-ones(22,1)]
song_filenames(:,2)=num2cell([-ones(22,1)]);

%% F_Test
for LOOP=9:22
    filename=song_filenames{LOOP};
    filename_numbers=size(filename,2); 
    filename=filename(1:filename_numbers-4);
    load(filename);
    input_for_test=T(:,2:25);
    output_for_test=T(:,1);
%导入特征值
    inputn_for_test=mapminmax('apply',input_for_test',inputs);
    result_for_test=sim(net,inputn_for_test);
    Result_for_test=mapminmax('reverse',result_for_test,outputs);
%处理特征值

%% F_Test
sum=0;
Er=0;
error=0;
for i=1:size(output_for_test,1)
    if Result_for_test(i)>=0.65
        Result_for_test(i)=1;
    end
    if Result_for_test(i)<0.65
        Result_for_test(i)=0;
    end
end  %将Result_for_test化为0\1
error=Result_for_test-output_for_test';
for i=1:size(output_for_test,1)
   if error(i)~=0
    sum=sum+1;
   end
end
Er=abs(1-sum/size(output_for_test,1)); %计算喜爱度
%list_for_sort=strcat(num2str(Er),'_',filename);
%A{LOOP-8}=list_for_sort;
%A=sortrows(A',-1);
song_filenames(LOOP,2)=num2cell(Er);
end
%% 排序
TJ=sortrows(song_filenames,-2);
%% 降序推荐歌曲
for i=1:14
    
    fprintf('第%d位推荐:喜爱度:%f%%  歌曲名:%s \n',i,cell2mat(TJ(i,2))*100,cell2mat(TJ(i,1)));
end




 
