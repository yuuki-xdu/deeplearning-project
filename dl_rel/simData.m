%% 直真君智，模拟724所的数据，通过信号特征进行分类识别
%1、信道号，2、编号，3、中频，4、载频，5、脉宽（单位s），6、方位（°），7、脉幅（dB），8、到达时间、9、到达时间差
%有用的只是4、5、6、7、8、和9 此处使用4，5，9用于深度学习 
%产生对应的数据，提供qt编程的数据
clc;clear all;close all
%频率范围10GHz~9GHz之间，SAR模式固定，或是有规律变化，其他模式无规律
%脉冲宽度，为与724的接近，选择6个值，分别为10、15、20、25、30、35、40，实际单位为μs，为与724的接近，使用整数值
%方位，按照某一个角度线性变换给出值，基本上可能不大用
%幅度，按照-10~-100之间，如果是扫描的按照扫描脉冲数给出大的值的变化
%到达时间，按照μs给出值，26.5μs的整数倍
%到达间隔，到达时间的前后差
%1、气象导航：载频、脉冲间隔无规律变化,幅度有周期，脉宽固定10，幅度有周期变化
%2、对海搜索：载频固定，脉冲间隔无规律变化，脉宽固定10，幅度有周期变化
%3、ISAR：  载频固定、脉冲间隔固定，脉宽固定20，幅度在约-60~-80之间随机变化
%4、扫描SAR：载频固定、脉冲间隔1k-1.7k之间固定，脉宽固定
%5、聚束SAR：载频固定、脉冲间隔滑变，脉宽固定，脉幅
%6、对潜：   载频固定，脉冲间隔固定、脉宽固定，搜索周期0.2s，脉幅周期变化
%有两种信号合并了，所以这个问题是5分类
% 定义生成数据的数量
n = 10000;                                          % 每种信号模式生成n个样本
N_mode=5;              %模式的数量（1、气象导航；2、对海搜索；3、ISAR；5、SAR；6、对潜探测）设定为0-4共5种作为标签
total_samples = 6 * N_mode; % 总样本数量
N=ones(1,N_mode)*n;%每种设置成n个脉冲
M=100;%假设因为频率和接收器器件问题带来的测量精度在100量级以内
PGB=26.5e-6;

%设置第一种，气象导航：载频、脉冲间隔无规律变化,幅度有周期，脉宽固定10，幅度有周期变化
%频率
f1=rand(1,N(1))+9;
F1=f1*1e9;
%脉冲宽度,固定为10
PW1=rand(1,N(1))-.5+10;
%脉冲间隔
PG1=floor(rand(1,N(1))*20)*26.5e-6;
%脉冲幅度
PA1=rand(1,N(1))*(-100)-60;

%设置第二种，对海搜索：载频固定，脉冲间隔无规律变化，脉宽固定10，幅度有周期变化
%频率
T2=32;%32个脉冲为一个扫描周期
NN=floor(N(2)/T2);
MM=N(2)-NN*T2;
f2=rand(1,T2)/M-rand(1,1)+10;
f2=f2*1e9;
%脉冲宽度10μs，不变
PW2=rand(1,N(2))/M+10;
%脉冲间隔无规律
pg2=floor(rand(1,T2)*200)+100;
pg2=pg2*26.5e-6;
%脉冲幅度有最大值周期出现
kk=floor(rand(1,1)*T2);
pa2=rand(1,T2)*(-100)-60;
pa2(kk)=max(pa2)+20;
F2=f2;
PG2=pg2;
PA2=pa2;
for k=2:NN
    F2=[F2,f2];
    PG2=[PG2,pg2];
    PA2=[PA2,pa2];
end
F2=[F2,f2(1:MM)];
PG2=[PG2,pg2(1:MM)];
PA2=[PA2,pa2(1:MM)];

%设置第三种，ISAR：载频固定、脉冲间隔固定，脉宽固定20，幅度在约-60~-80之间随机变化
f3=rand(1,N(3))/M-rand(1,1)+10;
F3=f3*1e9;
PG3=floor(rand(1,1)*20)*PGB*ones(1,N(3));
PW3=20+(rand(1,N(3))-0.5)/M;
PA3=(rand(1,N(3)))*20-80;

%设置第四种，扫描SAR：载频固定、脉冲间隔1k-1.7k之间固定
f4=rand(1,N(4))/M-rand(1,1)+10;
F4=f4*1e9;
pg4=rand(1,1)*700+100;
PG4=ones(1,N(4))/pg4;
PW4=rand(1,1)*40+rand(1,N(4))/M;
PA4=(rand(1,N(4)))*20-80; 
%原第五种和第四种合并了
% %设置第五种，聚束SAR：载频固定、脉冲间隔滑变，脉宽固定，脉幅
% f5=rand(1,N(5))/M-rand(1,1)+10;
% F5=f5*1e9;%载频固定
% %脉冲间隔滑变
% 
% %脉宽固定
% PW5=rand(1,1)*40+rand(1,N(5))/M;
% PA5=(rand(1,N(5)))*20-80;
%设置第5种，对潜：载频固定，脉冲间隔固定、脉宽固定，搜索周期0.2s，脉幅周期变化
f5=rand(1,N(5))/M-rand(1,1)+10;
F5=f5*1e9;
PW5=2.5+rand(1,N(5))/M;
PG5=floor(rand(1,1)*40)*PGB+rand(1,N(5))*1e-7/M;
MM=floor(0.2/mean(PG5));%每个扫描周期的脉冲数
NN=N(5)-floor(N(5)/MM)*MM;%剩余脉冲数
PA5=rand(1,N(5))*20-80;
pa5=floor(rand(1,1)*MM);%最大值出现的位置
PA5(pa5:MM:N(5))=max(PA5)+20;



clear f1 f2 f3 f4 f5 f6 pa1 pa2 pa3 pa4 pa5 pa6 pw1 pw2 pw3 pw4 pw5 pw6 
clear pg1 pg2 pg3 pg4 pg5 pg6 k kk M MM NN

PG=[PG1,PG2,PG3,PG4,PG5];
PGS(1)=PG(1);
for k=2:length(PG)
    PGS(k)=PGS(k-1)+PG(k);
end

K=sum(N);
%构造矩阵
%第五类先不进行仿真，测试一下其他部分有没有问题
%第一列，通道数255，第二列 3，第三类，9，
C1=ones(1,K)*1;
C2=ones(1,K)*2;
C3=ones(1,K)*3;
C4=[F1,F2,F3,F4,F5];
C5=[PW1,PW2,PW3,PW4,PW5];
C6=rand(1,K)+(1:K)/max(K)*30;
C7=[PA1,PA2,PA3,PA4,PA5];
C8=PGS;
C9=[PG1,PG2,PG3,PG4,PG5];
% 对应的标签0到5
labels1 = zeros(n, 1);    % 模式1的标签为0
labels2 = ones(n, 1);     % 模式2的标签为1
labels3 = ones(n, 1) * 2; % 模式3的标签为2
labels4 = ones(n, 1) * 3; % 模式4的标签为3
labels5 = ones(n, 1) * 4; % 模式5的标签为4
%只有5种模式
%labels6 = ones(n, 1) * 5; % 模式6的标签为5
C10=[labels1',labels2',labels3',labels4',labels5'];
data=[C1;C2;C3;C4;C5;C6;C7;C8;C9;C10]';%横向拼起来再转置
%在数据集生成过程中就进行打乱操作
rowrank = randperm(size(data, 1));      % size获得a的行数，randperm打乱各行的顺序
data1 = data(rowrank,:);                % 按照rowrank重新排列各行，注意rowrank的位置




% 保存为CSV文件，最后一个是标签列
csvwrite('signal_data_with_labels.csv', data1);
msgbox('complete');




% path='';%输入要保存的文件的路径
% writefile=[path,''];%输入要保存的文件名
% save(writefile,'data','N');
% msgbox('game over!')




% % 假设每种模式已经生成了N(1)到N(6)个脉冲数据
% % 在原有代码基础上，取每种模式前n个脉冲数据,目前设置成每种模式已经生成10k个，所以n应该小于等于10k
% F1 = F1(1:n); PW1 = PW1(1:n); PA1 = PA1(1:n); PG1 = PG1(1:n); PGS1 = PGS(1:n);
% F2 = F2(1:n); PW2 = PW2(1:n); PA2 = PA2(1:n); PG2 = PG2(1:n); PGS2 = PGS(1:n);
% F3 = F3(1:n); PW3 = PW3(1:n); PA3 = PA3(1:n); PG3 = PG3(1:n); PGS3 = PGS(1:n);
% F4 = F4(1:n); PW4 = PW4(1:n); PA4 = PA4(1:n); PG4 = PG4(1:n); PGS4 = PGS(1:n);
% F5 = F5(1:n); PW5 = PW5(1:n); PA5 = PA5(1:n); PG5 = PG5(1:n); PGS5 = PGS(1:n);
% F6 = F6(1:n); PW6 = PW6(1:n); PA6 = PA6(1:n); PG6 = PG6(1:n); PGS6 = PGS(1:n);



% % 合并数据，将每种模式的数据拼接在一起
% % 其中每一列分别为 1. 载频 2. 脉宽 3. 方位 4. 脉冲幅度 5. 到达时间 6. 到达时间差 7. 标签
% C1 = [F1', PW1', rand(1,n)', PA1', PGS1', PG1', labels1'; ...
%       F2', PW2', rand(1,n)', PA2', PGS2', PG2', labels2'; ...
%       F3', PW3', rand(1,n)', PA3', PGS3', PG3', labels3'; ...
%       F4', PW4', rand(1,n)', PA4', PGS4', PG4', labels4'; ...
%       F5', PW5', rand(1,n)', PA5', PGS5', PG5', labels5'; ...
%       F6', PW6', rand(1,n)', PA6', PGS6', PG6', labels6'];


