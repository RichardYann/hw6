clear;clc;
lena=imread('lena.bmp');
%% ��˹�����ͽ���������
lena_GauNoise_1=GaussianNoise(lena,0,0.1);
lena_GauNoise_2=GaussianNoise(lena,0,0.5);
lena_GauNoise_3=GaussianNoise(lena,0.5,0.1);
figure;
subplot(2,2,1);imshow(lena);title('ԭʼͼ��');
subplot(2,2,2);imshow(lena_GauNoise_1);title('�Ӹ�˹������ͼ��(av=0,std=0.1)');
subplot(2,2,3);imshow(lena_GauNoise_2);title('�Ӹ�˹������ͼ��(av=0,std=0.5)');
subplot(2,2,4);imshow(lena_GauNoise_3);title('�Ӹ�˹������ͼ��(av=0.5,std=0.1)');

lena_SPNoise_1=SaltPepperNoise(lena,0.1,0.1);
lena_SPNoise_2=SaltPepperNoise(lena,0.1,0);
lena_SPNoise_3=SaltPepperNoise(lena,0,0.1);
figure;
subplot(2,2,1);imshow(lena);title('ԭʼͼ��');
subplot(2,2,2);imshow(lena_SPNoise_1);title('�ӽ���������ͼ��(k1=k2=0.1)');
subplot(2,2,3);imshow(lena_SPNoise_2);title('�Ӻ���������ͼ��(k1=0.1,k2=0)');
subplot(2,2,4);imshow(lena_SPNoise_3);title('�Ӱ���������ͼ��(k1=0,k2=0.1)');

%% �������������˲����ָ���
lena_GN_Gau=GaussianFilter(lena_GauNoise_1,5,1.5);
lena_GN_Med=MedianFilter(lena_GauNoise_1,5);
figure;
subplot(2,2,1);imshow(lena);title('ԭʼͼ��');
subplot(2,2,2);imshow(lena_GauNoise_1);title('�Ӹ�˹������ͼ��');
subplot(2,2,3);imshow(lena_GN_Gau);title('��˹�˲���ͼ��');
subplot(2,2,4);imshow(lena_GN_Med);title('��ֵ�˲���ͼ��');
lena_SN_Gau=GaussianFilter(lena_GauNoise_1,5,1.5);
lena_SN_Med=MedianFilter(lena_GauNoise_1,5);
figure;
subplot(2,2,1);imshow(lena);title('ԭʼͼ��');
subplot(2,2,2);imshow(lena_SPNoise_1);title('�ӽ���������ͼ��');
subplot(2,2,3);imshow(lena_SN_Gau);title('��˹�˲���ͼ��');
subplot(2,2,4);imshow(lena_SN_Med);title('��ֵ�˲���ͼ��');

%% Q�������Խ������������ã�
lena_SPNoise1=SaltPepperNoise(lena,0.1,0);
lena_SPNoise2=SaltPepperNoise(lena,0,0.1);
lena_SN1_Con1=Contraharmonic(lena_SPNoise1,1.5);
lena_SN1_Con2=Contraharmonic(lena_SPNoise1,-1.5);
lena_SN2_Con1=Contraharmonic(lena_SPNoise2,1.5);
lena_SN2_Con2=Contraharmonic(lena_SPNoise2,-1.5);
figure;
subplot(3,2,1);imshow(lena_SPNoise1);title('�Ӻ�������');
subplot(3,2,2);imshow(lena_SPNoise2);title('�Ӱ�������');
subplot(3,2,3);imshow(lena_SN1_Con1);title('Q=1.5�Ժ�������');
subplot(3,2,4);imshow(lena_SN1_Con2);title('Q=-1.5�Ժ�������');
subplot(3,2,5);imshow(lena_SN2_Con1);title('Q=1.5�԰�������');
subplot(3,2,6);imshow(lena_SN2_Con2);title('Q=-1.5�԰�������');

%% ģ��&�˶�ģ����
lena_blur=Blur(lena,0.1,0.1,1);
lena_blur_Gau=GaussianNoise(lena_blur,0,0.1);
lena_MotionBlur_H=fspecial('motion',50,45);
lena_MotionBlur=imfilter(lena,lena_MotionBlur_H,'circular','conv');
lena_MotionBlur_Gau=GaussianNoise(lena_MotionBlur,0,0.1);

figure;
subplot(2,2,1);imshow(lena);title('ԭͼ');
subplot(2,2,2);imshow(lena_blur);title('ģ����ͼ��');
subplot(2,2,3);imshow(lena_blur_Gau);title('ģ���Ӹ�˹����');

figure;
subplot(2,2,1);imshow(lena);title('ԭͼ');
subplot(2,2,2);imshow(lena_MotionBlur);title('�˶�ģ����ͼ��');
subplot(2,2,3);imshow(lena_MotionBlur_Gau);title('�˶�ģ���Ӹ�˹����');

%% ά���˲�����
lena_motion_H=fspecial('motion',50,45);
lena_motionblur=imfilter(lena,lena_motion_H,'circular','conv');
lena_motionblur_Gau=imnoise(lena_motionblur,'gaussian',0,0.01);
%ά���˲���
noise=imnoise(zeros(size(lena)),'gaussian',0,0.01);
NSR=sum(noise(:).^2)/sum(im2double(lena(:)).^2);
lena_Wiener=deconvwnr(lena_motionblur_Gau,lena_motion_H,NSR);
%Լ����С�����˲���
V=0.0001;
NoisePower=V*numel(lena);
[lena_CLS,LAGRA]=deconvreg(lena_motionblur,lena_motion_H,NoisePower);
%���ͼ��
figure;
subplot(2,2,1);imshow(lena_motionblur);title('�˶�ģ��ͼ��');
subplot(2,2,2);imshow(lena_motionblur_Gau);title('���벢ģ��ͼ��');
subplot(2,2,3);imshow(lena_Wiener);title('ά���˲��Ľ��');
subplot(2,2,4);imshow(lena_CLS);title('Լ����С�����˲�');

%% *************** ��   �� *****************
%�Ӹ�˹������
function Img_out=GaussianNoise(Img,av,std)
[M,N]=size(Img);
u1=rand(M,N);   u2=rand(M,N);
x=std*sqrt(-2*log(u1)).*cos(2*pi*u2)+av;
Img_out=uint8(255*(double(Img)/255+x));
end
%�ӽ���������
function Img_out=SaltPepperNoise(Img,a,b)
[M,N]=size(Img);
x=rand(M,N);
Img_out=Img;
Img_out(find(x<=a))=0;
Img_out(find(x>a&x<(a+b)))=255;
end
% ��˹�˲���
function Img_out=GaussianFilter(Img,masksize,sigma)
for i=1:masksize
    for j=1:masksize
        x=i-ceil(masksize/2);
        y=j-ceil(masksize/2);
        h(i,j)=exp(-(x^2+y^2)/(2*sigma^2))/(2*pi*sigma^2);
    end
end
Img_out=uint8(conv2(Img,h,'same'));
end
% ��ֵ�˲���
function Img_out=MedianFilter(Img,masksize)
exsize=floor(masksize/2);   %��������չ��С
Imgex=padarray(Img,[exsize,exsize],'replicate','both'); %��չͼƬ
[m,n]=size(Img);
Img_out=Img;    %��Img_out׼��Ϊ��Img��ͬ��size
for i=1:m
    for j=1:n
        neighbor=Imgex(i:i+masksize-1,j:j+masksize-1);  %��ȡ����
        Img_out(i,j)=median(neighbor(:));   %��ֵ�˲�
    end
end
end
%��г����ֵ�˲���
function Img_out=Contraharmonic(Img,Q)
[M,N]=size(Img);
ImgSize=3;   ImgSize=(ImgSize-1)/2;
Img_out=Img;
for x=1+ImgSize:1:M-ImgSize
    for y=1+ImgSize:1:M-ImgSize
        is=Img(x-ImgSize:1:x+ImgSize,y-ImgSize:1:y+ImgSize);
        Img_out(x,y)=sum(double(is(:)).^(Q+1))/sum(double(is(:)).^(Q));
    end
end
end
%ģ����
function Img_out=Blur(Img_in,a,b,T)
Img_in=double(imread('lena.bmp'));
Img_fft_shift=fftshift(fft2(Img_in));
[M,N]=size(Img_fft_shift);
for i=1:M
    for j=1:N
        H(i,j)=(T/(pi*(i*a+j*b)))*sin(pi*(i*a+j*b))*exp(-sqrt(-1)*pi*(i*a+j*b));
        G(i,j)=H(i,j)*Img_fft_shift(i,j);
    end
end
Img_out=ifft2(ifftshift(G));
Img_out=256.*Img_out./max(max(Img_out));
Img_out=uint8(real(Img_out));
end