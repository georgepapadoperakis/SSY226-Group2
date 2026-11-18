"""Direkt översättning av de packade fälten i estimateParameters.m.
Stöder vektoriserad beräkning. Python/MATLAB-paritet är ännu inte verifierad.
Bevarar originalets fasta uteluftsegenskaper vid 263.15 K.
"""
import numpy as np
from CoolProp.CoolProp import PropsSI

def parameters(x, ambient, speed=0., ib=0., heat=(0.,0.,0.), fan=0., ctrl=(0.,1.,0.,0.,0.,0.,0.)):
    x = np.asarray(x, dtype=float)
    lo,hi,air=x[...,5],x[...,6],x[...,8]
    def sat(key,p,q):return PropsSI(key,'P',p,'Q',q,'R134a')
    def gas(key,t):return PropsSI(key,'T',t,'P',101325.,'Air')
    d={}
    d['envT']=ambient+273.15
    for name,val in [('Cp',2430.),('K',.254),('Rho',1113.),('Mu',.0161)]:d['clnt'+name]=val
    for pre,t in [('cabAir',air),('condEvapAir',263.15)]:
        for name,key in [('Cp','C'),('K','L'),('Rho','D'),('Mu','V')]:d[pre+name]=gas(key,t)
    for pre,p,q in [('innerCondRef',hi,1),('condEvapRef',lo,0),('evapRef',lo,0),('chilRef',lo,0)]:
        for name,key in [('Cp','C'),('K','L'),('Rho','D'),('Mu','V')]:d[pre+name]=sat(key,p,q)
    hin=PropsSI('H','P',lo,'T',sat('T',lo,1)+2.,'R134a')
    sin=PropsSI('S','P',lo,'H',hin,'R134a')
    d.update(speed=speed,compHin=hin,compHoutS=PropsSI('H','P',hi,'S',sin,'R134a'),
        compVnominalIn=1/PropsSI('D','P',3e5,'T',284.15,'R134a'),
        refTsatHighPK=sat('T',hi,1),refCpVapHighP=sat('C',hi,1),
        refHsatLiqHighP=sat('H',hi,0),refTsatLiqHighPK=sat('T',hi,0),
        refTsatLiqLowPK=sat('T',lo,0),motQmeas=heat[0],invQmeas=heat[1],dcdcQmeas=heat[2],
        evapVoidFrac=.8,condVoidFrac=.5,evapVol_m3=.001,condVol_m3=.001,
        evapWallThermalMass_J_K=5000.,condWallThermalMass_J_K=5000.,
        batImeas=ib,refCpLiqHighP=sat('C',hi,0),fanRpm_prev=fan)
    for p,tname,suffix in [(lo,'dTesat_dp1','dp1'),(hi,'dTcsat_dp2','dp2')]:
        d[tname]=(sat('T',p+100,1)-sat('T',p-100,1))/200.
        for q,phase in [(0,'l'),(1,'g')]:
            d[f'd_rhoh_{phase}_{suffix}']=(sat('D',p+100,q)*sat('H',p+100,q)-sat('D',p-100,q)*sat('H',p-100,q))/200.
    names=['cmdHeater','cmdParallelSerial','cmdBlendAir','cmdExvEvap','cmdExvChiller','thermalMode','cmdExvHeatpump']
    d.update(zip(names,ctrl))
    return d
