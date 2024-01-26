clc;
% clear all; 
 close all;
% Sample_Num=36;
% Sample_data=500;
% Channels=11;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% N=500; %%% max time epoch
% dt=0.002;
% % Excitatory neurons    Inhibitory neurons
% Ne=72;                 Ni=28;
% Nn=Ne+Ni;
% re=rand(Ne,1);          ri=rand(Ni,1);
% a=[0.02*ones(Ne/2,1);0.02+0.08*ri(1:Ni/2,1);0.02*ones(Ne/2,1);  0.02+0.08*ri(1+Ni/2:Ni,1) ];
% b=[0.2*ones(Ne/2,1);      0.25-0.05*ri(1:Ni/2,1);0.2*ones(Ne/2,1);      0.25-0.05*ri(1+Ni/2:Ni,1) ];
% c=[-65+15*re(1:Ne/2,1).^2;        -65*ones(Ni/2,1);-65+15*re(Ne/2+1:Ne,1).^2;        -65*ones(Ni/2,1)];
% d=[8-6*re(1:Ne/2,1).^2;           2*ones(Ni/2,1); 8-6*re(Ne/2+1:Ne,1).^2;           2*ones(Ni/2,1)];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% ef=[];
% woi=zeros(Nn);
% 
% 
%  wm=zeros(N,Nn,Nn);
% dw=zeros(Nn,Nn);
% w=zeros(Nn,Nn);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% formatSpec = "EEG_sample%d.xlsx";
% M=cell(Sample_Num,1);
% kk=1;
% for k=1:Sample_Num
% filename=compose(formatSpec,k)
% u=readmatrix(filename);
% 
% M(k)={u'};
% 
% end
% % % % % % % % % % % % % % % % % % % % % % % % 
Id=random('norm',0,4,Nn,N);
I=random('norm',0,6,Nn,N);
V=zeros(Nn,N);
V(:,1)=-95*ones(Nn,1);    % Initial values of v
u=zeros(Nn,N);   
u(:,1)=b.*V(:,1);% Initial values of u
dV=zeros(Nn,N);

% % % % % % % % % % % % % % % % 

% % % % % % % % % % % % % % % % % % % % % % % 
Ins=random('norm',0,2,Nn,N);
Id=-5*10^(-6)*ones(Nn,1);
Isyn=zeros(Nn,N);

V=zeros(Nn,N);
dV=zeros(Nn,N);
ddV=zeros(Nn,N); %% first Nm +second Nq is arrenged= Nn
u=randn(Nn,N);
du=zeros(Nn,N);
%%%%%%%%%%%% input EEG
% input--------------------------------------------------------------input
X=zeros(Sample_Num,Channels);

for samp=1:Sample_Num
X=cell2mat(M(samp));
Nm=size(X,1);

%%%%%%%%%%%%%%%%%%%%%%%%%
Sd=[];So=[];Si=[];
% Sd=Si;
inp=X;
% % clear X;
delta=zeros(Nn,N);
Xx=zeros(Nn,N);
inx=[];indd=[];
% % % % % % % % % % % % % % % 
taud=5000;t=0; So=Xx;
ad=1;Adi=0.1;taudi=5000;Aid=0.1;tauid=5000;
Aoi=Adi;tauoi=2;ado=1;dwoi=rand(Nn);

clear f_inp f_out inx inxh indd
f_inp=Xx;
f_out=Xx;
m_in=-0.5;std_in=1;
m_v=-0.3;std_v=1;
xd=[];
xd=[xd,zeros(Channels,1)];
% % % % % % % % % % % % % % % % % % % % % % % % % 
for i=1:N-1
    t=t+dt;
    
[Xk,m_in,std_in,m_v,std_v]=NewSpikeDetection2(inp(:,i+1),inp(:,i),dt,i,m_in,std_in,m_v,std_v);
%
xd=[xd,Xk];
 w=w+0.9*(dw+dwoi);
 % % %      Neuron Model / Izhikevich 2003
 
  dV(:,i)=0.04*V(:,i).*V(:,i)+5*V(:,i)+140*ones(Nn,1)-u(:,i)+Ins(:,i)+10*Isyn(:,i)+Id;
%   dV(:,i)=0.04*V(:,i).*V(:,i)+5*V(:,i)+140*ones(Nn,1)-u(:,i)+100*Isyn(:,i);
  
%   dV(1:Nm,i)=dV(1:Nm,i)+100*Isyn(1:Nm,i);
  du(:,i)=a.*(b.*V(:,i)-u(:,i));
  V(:,i+1)=V(:,i)+dV(:,i)*dt;
  u(:,i+1)=u(:,i)+du(:,i);
  for j=1:Nn
     if V(j,i+1)>30
       V(j,i+1)=c(j);
       u(j,i+1)=u(j,i+1)+d(j);
       Xx(j,i+1)=1;
     else
         Xx(j,i+1)=0;
     end
  end
  f_out(:,i)=Xx(:,i)*dt*i;
  f_inp(1:Nm,i)=Xk*dt*i;
%    f_inp(Nm+1:end,i)=Xx(Nm+1:end,i);
   
    if i>=2
        f_inp(find(f_inp(:,i)==0),i) =f_inp(find(f_inp(:,i)==0),i-1);
    end
    Sd(:,i)=f_inp(:,i);
    Si(:,i)=f_inp(:,i);
   f_out(:,i+1)=i*dt*Xx(:,i+1);
  if i>2
     So(find(f_out(:,i)==0),i)=So(find(f_out(:,i)==0),i-1); 
  end
  s=Si(:,i)-Sd(:,i);
  for j=1:Nn
     if s(j)<=0
         adi(j)=Adi*exp(s(j)/taudi);
     else 
         adi(j)=Aid*exp(-s(j)/tauid);
     end
     
     for jj=1:Nm
     dwoi(jj,j)=(inp(jj,i)-Xx(jj,i))*(ad+conv(adi(j),inp(jj,i)))';
     end
     for jj=Nm+1:Nn
         if i<=N-1
     dwoi(jj,j)=adi(j);
         end
     end
    
  end
   woi=woi+dwoi;
   Isyn(:,i+1)=woi*(exp(t/taud)*ones(Nn,1));
  A11=filt70(inp(:,i));
A12=filt70(Xx(1:Nm,i));
    ef(:,i)=(A11'*A12)/(norm(A11,2)*norm(A12,2));
end
% subplot(3,1,1)
% plot(1:N,V(2,:)')
% title('output')
% subplot(3,1,2)
% stem(1:N,Xx(2,:)')
% title('out spike')
% subplot(3,1,3)
% stem(1:N,X(:,2))
% title('desiered')

figure(1);
plot(1:N-1,ef)
title('performance of RSM with process noise')
ylim([0 1.5]);
xlabel('Samples');
ylabel('Amplitude');
indd=zeros(Nm,N);
   for j=1:Nm
   for i=1:N
       if xd(j,i)==1
      indd(j,i)= j;
           
       end

   end
   end
    for j=1:Nm
   for i=1:N
      if Xx(j,i)==1
      inx(j,i)= j;
           
       end
   end
    end 
       
%    xlim([0 700]);
   figure(2);
   subplot(2,1,2)
   plot(inx',' . ')
   title('Observed Neurons firing pattern using RSM with process noise','Fontsize',8);
   
   xlabel('Samples')
   ylabel('Neuron number')
%    ylim([1 Nm])
   ylabel('Neuron number')
   lb={'Fp1','Fp2','C3','C4','O1','O2','F7','F8','T3','T6','Cz'};
   legend(lb)
   xlim([0 100]);
%     figure;
subplot(2,1,1)
  plot(indd','.')
   title('Encoded firing pattern of sample EEG Channels ','Fontsize',9);
     xlabel('Samples')
   ylabel('Neuron number')
   ylim([1 Nm]);
    lb={'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'};
   legend(lb)
   xlim([0 100]);
   if samp<=6
   title('RSM firing pattern output-Epilepsy');
   else if samp<22
         title('RSM firing pattern output-Migraine');   
       else
           title('RSM firing pattern output-Normal');   
       end
   end
   xlabel('time(sec)')
   xlim([0 100])
   ylabel('Neuron Number')
   fsc='firing rate-RSM%d.tif';
  fname=compose(fsc,samp);
 ff=cell2mat(fname);
   saveas(gcf,ff)
   
   
        figure(3);
for j=Nm+1:Nn
   for i=1:N
       
    if Xx(j,i)==1
      inxh(j,i)= j;
           
       end
   end
      end
      plot(inxh',' . ')
%    title('RSM   neuronal firing pattern of sample EEG Channels ','Fontsize',9);
    xlabel('Samples')
    ylabel('Neuron Number')
%      xlim([0 700]);
    ylim([Nm+1 Nn])
    if samp<=6
   title('RSM firing pattern Hidden Neurons-Epilepsy');
   else if samp<22
         title('RSM firing pattern  Hidden Neurons-Migraine');   
       else
           title('RSM firing pattern  Hidden Neurons-Normal');   
       end
   end
   xlabel('time(sec)')
   xlim([0 100])
   ylabel('Neuron Number')
   fsc=' Hidden firing rate-RSM%d.tif';
   fname=compose(fsc,samp);
   ff=cell2mat(fname);
   saveas(gcf,ff)
end
%     addpath 'C:\Users\Samaneh\Documents\PHD-thesis2022\papers\paper2\P300-BCI-main\P300-BCI-main\codes'
% 
%      [n_correct0, n_test0] = method_deep_cnn(M(1:25,:,:), M(26:33,:,:), {'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'})
% %