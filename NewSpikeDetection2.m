function  [Xk,m_in,std_in,m_v,std_v]=NewSpikeDetection(V1,V2,dt,kk,m_in0,std_in0,m_v0,std_v0)
% V1 is V of the Nuron V2 is the pervious V
Vx=(V1-V2)./dt;
m_in=((kk-1)*m_in0+Vx)./kk;
std_in=((kk-1)*std_in0+abs(Vx-std_in0))./kk;
Vnx=(Vx-m_in0)./std_in0;
m_v=((kk-1)*m_v0+Vnx)./kk;
std_v=((kk-1)*std_v0+abs(Vnx-std_v0))./kk;
Xk=Vnx>=(m_v+std_v/2);


end