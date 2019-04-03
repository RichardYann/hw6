clear;clc;
lena=imread('lena.bmp');
%% 高斯噪声和椒盐噪声：
lena_GauNoise_1=GaussianNoise(lena,0,0.1);
lena_GauNoise_2=GaussianNoise(lena,0,0.5);
lena_GauNoise_3=GaussianNoise(lena,0.5,0.1);
figure;
subplot(2,2,1);imshow(lena);title('原始图像');
subplot(2,2,2);imshow(lena_GauNoise_1);title('加高斯噪声后图像(av=0,std=0.1)');
subplot(2,2,3);imshow(lena_GauNoise_2);title('加高斯噪声后图像(av=0,std=0.5)');
subplot(2,2,4);imshow(lena_GauNoise_3);title('加高斯噪声后图像(av=0.5,std=0.1)');

lena_SPNoise_1=SaltPepperNoise(lena,0.1,0.1);
lena_SPNoise_2=SaltPepperNoise(lena,0.1,0);
lena_SPNoise_3=SaltPepperNoise(lena,0,0.1);
figure;
subplot(2,2,1);imshow(lena);title('原始图像');
subplot(2,2,2);imshow(lena_SPNoise_1);title('加椒盐噪声后图像(k1=k2=0.1)');
subplot(2,2,3);imshow(lena_SPNoise_2);title('加胡椒噪声后图像(k1=0.1,k2=0)');
subplot(2,2,4);imshow(lena_SPNoise_3);title('加白盐噪声后图像(k1=0,k2=0.1)');

%% 加入噪声并用滤波器恢复：
lena_GN_Gau=GaussianFilter(lena_GauNoise_1,5,1.5);
lena_GN_Med=MedianFilter(lena_GauNoise_1,5);
figure;
subplot(2,2,1);imshow(lena);title('原始图像');
subplot(2,2,2);imshow(lena_GauNoise_1);title('加高斯噪声后图像');
subplot(2,2,3);imshow(lena_GN_Gau);title('高斯滤波后图像');
subplot(2,2,4);imshow(lena_GN_Med);title('中值滤波后图像');
lena_SN_Gau=GaussianFilter(lena_GauNoise_1,5,1.5);
lena_SN_Med=MedianFilter(lena_GauNoise_1,5);
figure;
subplot(2,2,1);imshow(lena);title('原始图像');
subplot(2,2,2);imshow(lena_SPNoise_1);title('加椒盐噪声后图像');
subplot(2,2,3);imshow(lena_SN_Gau);title('高斯滤波后图像');
subplot(2,2,4);imshow(lena_SN_Med);title('中值滤波后图像');

%% Q的正负对椒盐噪声的作用：
lena_SPNoise1=SaltPepperNoise(lena,0.1,0);
lena_SPNoise2=SaltPepperNoise(lena,0,0.1);
lena_SN1_Con1=Contraharmonic(lena_SPNoise1,1.5);
lena_SN1_Con2=Contraharmonic(lena_SPNoise1,-1.5);
lena_SN2_Con1=Contraharmonic(lena_SPNoise2,1.5);
lena_SN2_Con2=Contraharmonic(lena_SPNoise2,-1.5);
figure;
subplot(3,2,1);imshow(lena_SPNoise1);title('加胡椒噪声');
subplot(3,2,2);imshow(lena_SPNoise2);title('加白盐噪声');
subplot(3,2,3);imshow(lena_SN1_Con1);title('Q=1.5对胡椒噪声');
subplot(3,2,4);imshow(lena_SN1_Con2);title('Q=-1.5对胡椒噪声');
subplot(3,2,5);imshow(lena_SN2_Con1);title('Q=1.5对白盐噪声');
subplot(3,2,6);imshow(lena_SN2_Con2);title('Q=-1.5对白盐噪声');

%% 模糊&运动模糊：
lena_blur=Blur(lena,0.1,0.1,1);
lena_blur_Gau=GaussianNoise(lena_blur,0,0.1);
lena_MotionBlur_H=fspecial('motion',50,45);
lena_MotionBlur=imfilter(lena,lena_MotionBlur_H,'circular','conv');
lena_MotionBlur_Gau=GaussianNoise(lena_MotionBlur,0,0.1);

figure;
subplot(2,2,1);imshow(lena);title('原图');
subplot(2,2,2);imshow(lena_blur);title('模糊后图像');
subplot(2,2,3);imshow(lena_blur_Gau);title('模糊加高斯噪声');

figure;
subplot(2,2,1);imshow(lena);title('原图');
subplot(2,2,2);imshow(lena_MotionBlur);title('运动模糊后图像');
subplot(2,2,3);imshow(lena_MotionBlur_Gau);title('运动模糊加高斯噪声');

%% 维纳滤波器：
lena_motion_H=fspecial('motion',50,45);
lena_motionblur=imfilter(lena,lena_motion_H,'circular','conv');
lena_motionblur_Gau=imnoise(lena_motionblur,'gaussian',0,0.01);
%维纳滤波：
noise=imnoise(zeros(size(lena)),'gaussian',0,0.01);
NSR=sum(noise(:).^2)/sum(im2double(lena(:)).^2);
lena_Wiener=deconvwnr(lena_motionblur_Gau,lena_motion_H,NSR);
%约束最小二乘滤波：
V=0.0001;
NoisePower=V*numel(lena);
[lena_CLS,LAGRA]=deconvreg(lena_motionblur,lena_motion_H,NoisePower);
%输出图像：
figure;
subplot(2,2,1);imshow(lena_motionblur);title('运动模糊图像');
subplot(2,2,2);imshow(lena_motionblur_Gau);title('加噪并模糊图像');
subplot(2,2,3);imshow(lena_Wiener);title('维纳滤波的结果');
subplot(2,2,4);imshow(lena_CLS);title('约束最小二乘滤波');

%% *************** 函   数 *****************
%加高斯噪声：
function Img_out=GaussianNoise(Img,av,std)
[M,N]=size(Img);
u1=rand(M,N);   u2=rand(M,N);
x=std*sqrt(-2*log(u1)).*cos(2*pi*u2)+av;
Img_out=uint8(255*(double(Img)/255+x));
end
%加椒盐噪声：
function Img_out=SaltPepperNoise(Img,a,b)
[M,N]=size(Img);
x=rand(M,N);
Img_out=Img;
Img_out(find(x<=a))=0;
Img_out(find(x>a&x<(a+b)))=255;
end
% 高斯滤波：
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
% 中值滤波：
function Img_out=MedianFilter(Img,masksize)
exsize=floor(masksize/2);   %各方向扩展大小
Imgex=padarray(Img,[exsize,exsize],'replicate','both'); %扩展图片
[m,n]=size(Img);
Img_out=Img;    %将Img_out准备为和Img相同的size
for i=1:m
    for j=1:n
        neighbor=Imgex(i:i+masksize-1,j:j+masksize-1);  %截取邻域
        Img_out(i,j)=median(neighbor(:));   %中值滤波
    end
end
end
%反谐波均值滤波：
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
%模糊：
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