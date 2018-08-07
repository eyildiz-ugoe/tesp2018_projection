import bge
import math
import time


def setup():

def main():
    setup()

    orbitalPeriod = [88.0,224.7,365.2,687.0,4331,10747,30589,59800]
    #orbitalPeriod = [88.0,224.7,365.2,687,433,1074,3058,5980]
    # = [int(x) for x in orbitalPeriod]
    orbitalPeriod[4:] = [x*0.005  for x in orbitalPeriod[4:]]
    orbitalVel = [47.4,35.0,29.8,24.1,13.1,9.7,6.8,5.4]
    orbitalVelP = orbitalVel[::-1]
    orbitalVelP = [x*0.5  for x in orbitalVelP]
    orbitalVelS = [10.8791667, 1]

    ORBITAL_PERIOD = [88.0,224.7,365.2,687.0,4331,10747,30589,59800]
    #orbitalPeriod = [88.0,224.7,365.2,687,433,1074,3058,5980]
    # = [int(x) for x in orbitalPeriod]
    ORBITAL_PERIOD[4:] = [x*0.01  for x in ORBITAL_PERIOD[4:]]



    scene = bge.logic.getCurrentScene()

    objL = []
    for ob in scene.objects:
        if 'pivot' in ob.name:
            objL.append(ob)
    #print(objL)
    '''
    for idx, x in enumerate(objL.reverse()):
        print(x.name)
        x.applyRotation([0.0,0.0,orbitalVel[idx]*0.01],True)
    '''
    #print(orbitalVel)
    idP = 0
    idS = 0
    oblP = []
    for x in objL:
        if 'P_' in x.name:
        #print(x.name)
            x.applyRotation([0.0,0.0,orbitalVelP[idP]*0.001],True)
            idP += 1
            oblP.append(x)
        elif 'S_' in x.name:
        #print(x.name)
            x.applyRotation([0.0,0.0,orbitalVelS[idS]*0.001],True)
            idS += 1
        #print(orbitalVel[idx])
    #time.sleep(0.1)

    for ob in scene.objects:
        nn = ob.name
        if 'P_' in nn and 'pivot' not in nn:
            ob.applyRotation([0.0,0.0,ORBITAL_PERIOD[idS]],True)

    keyboard= bge.logic.keyboard
    if bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.UPARROWKEY]:
        bge.render.makeScreenshot('/mnt/NewVolume/studia/s10/TESP/blend/Screenshot#.png')
        objL = []
        dataL = []
        xy = []
        for ob in scene.objects:
            nn = ob.name
            if 'P_' in nn and 'pivot' not in nn:
                #print(ob.worldPosition)
                #print(ob.name)
                #objL.append(ob)
                dataL.append([ob.name[2:], ob.worldPosition[0], ob.worldPosition[1], ob['Diam']])#radius
            #elif 'S_' in nn and 'pivot' not in nn:
                #print(ob.worldPosition)
                #print(ob.name)
                #objL.append(ob)
            #    dataL.append([ob.name[2:], ob.worldPosition[0], ob.worldPosition[1]])
        print(dataL)

if __name__ == "__main__":
    main()
