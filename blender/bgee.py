import bge
import math
import time

#storage = bge.logic.globalDict['Pipe']

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
ORBITAL_PERIOD[4:] = [x*0.001  for x in ORBITAL_PERIOD[4:]]

SLOW = 1

def lel(w):
    pass

def setup():
    own = bge.logic.getCurrentController().owner

    if 'init' not in own: # Will only run once, or when the var gets removed
        own['init'] = True
        from multiprocessing import Process, Pipe
        parent_pipe, child_pipe = Pipe()
        proc = Process(target=lel, args=(child_pipe,))
        proc.start()
        scene = bge.logic.getCurrentScene()
        lamp = scene.objects['Lamp']
        lamp['SL'] = 1
        lamp['proc'] = proc
        lamp['pipe'] = parent_pipe
        #print('only once')
        #scene = bge.logic.getCurrentScene()
        #global TEST
        #TEST="Gyvfvyfgvyfgv"
        #print('first')
        #for ob in scene.objects:
        #    nn = ob.name
        #    if 'P_' in nn and 'pivot' not in nn:
        #        make_texture(ob)

def mouse_hit_ray(mouse_x, mouse_y, property="",distance=200.0):
    rend = bge.render
    width = rend.getWindowWidth()
    height = rend.getWindowHeight()
    xx = int(width/2)
    yy = int(height/2)
    rend.setMousePosition( mouse_x, mouse_y)

    scene = bge.logic.getCurrentScene()
    camera = scene.active_camera
    return_normal = 0
    x_ray = 1
    return_polygon = 0

    #mouse_position = bge.logic.mouse.position
    #print(mouse_position)
    screen_vect = camera.getScreenVect(mouse_position[0],mouse_position[1])
    screen_vect.negate()
    target_position = camera.worldPosition + screen_vect
    target_ray = camera.rayCast( target_position, camera, distance , property)

    return target_ray

def send_to(data):
    scene = bge.logic.getCurrentScene()
    lamp = scene.objects['Lamp']
    lamp['pipe'].send(data)

def receive_from():
    scene = bge.logic.getCurrentScene()
    lamp = scene.objects['Lamp']
    try:
        a= lamp['pipe'].recv()
        return a
    except:
        return None


def make_texture(obj):
    #for x in objects:

    o = bpy.context.active_object
    o.select = True
    new_mat = bpy.data.materials.new(name="a")
    new_mat.diffuse_color = (0.8,0.8,0.0)
    o.data.materials.append(new_mat)



    matID = bge.texture.materialID(obj, obj.name)
    img = bge.texture.Texture(obj, matID)
    imagePath = bge.logic.expandPath('//'+ obj.name + '.png')
    image = bge.texture.ImageFFmpeg(imagePath)
    # load image into a bgl buffer
    image_buffer = bge.texture.imageToArray(image, 'RGB')
    # get image size
    image_size = image.size
    # use ImageBuff as the source
    img.source = bge.texture.ImageBuff()
    # load the image from the buffer into ImageBuff
    img.source.load(image_buffer, image_size[0], image_size[1])
    # get the image data
    data = img.source.image
    # save as an object variable
    obj["img"] = img
    # display the image
    img.refresh(True)

def follow(obj):
    scene = bge.logic.getCurrentScene()


    for ob in scene.objects:
        if 'empty' in ob.name and obj.name in ob.name:
            ori = ob.worldOrientation.to_euler()
            nn = ob.name
            #ori = ob.getRotation()
            '''
            print("st1")
            print(obj.position)
            print(obj.worldOrientation)
            print(obj.worldOrientation.to_euler())
            '''
            camera = scene.active_camera
            #print(camera.worldOrientation.to_euler())

            #camera.setParent(ob)
            #cpos = ob.worldPosition
            #camLocPos = [cpos[0]+2.5,cpos[1]-4, cpos[2]+1.5] # - obj.worldPosition
            #camera.worldPosition = [ob.worldPosition[0] + camLocPos[0], ob.worldPosition[1] + camLocPos[1], ob.worldPosition[2] + camLocPos[2]]
            camera.worldPosition = ob.worldPosition
            camera.worldOrientation = ob.worldOrientation

        #    opos = ob.worldPosition
            #opos[0 ]= ob.worldPosition[0]+2.5
            #opos[1] = ob.worldPosition[1]-4
            #opos[2] = 1.5
            #camera.worldPosition = opos
            #print(camera.worldOrientation.to_euler())
            #camera.setParent(ob)
            #camera.worldPosition = [obj.position[0]+2.5,obj.position[1]-4,1.5] # set camera position
            #camera.localPosition = [obj.position[0]+2.5,obj.position[1]-4,1.5] # set camera position
            #camera.applyMovement([obj.position[0]+2.5,obj.position[1]-4,1.5], True)
            #camera.worldPosition = [obj.position[0],obj.position[1]-7,1.2]
            #camera.localOrientation = [math.radians(60.0), math.radians(71.0), 0.0] # set camera orientation
            #camera.localOrientation = [0,0,0]
            #camera.setParent(ob)
            #oloc = ob.worldOrientation.to_euler()
            #camera.worldOrientation = -1*oloc
            #oloc = -1*oloc
            #m = camera.worldOrientation.to_euler()
            #m = [math.radians(45),math.radians(45), math.radians(45)]
            #m[2] = math.radians(120)
            #camera.worldOrientation = m.to_matrix()
            camera.setParent(ob)
            #print(camera.worldOrientation.to_euler())
            #camera.worldOrientation = [math.radians(80.0),  0.0, math.radians(71.0)]

            #camera.setParent(ob)
            '''
            print(camera.position)
            print(camera.localOrientation.to_euler())
            print(camera.worldOrientation.to_euler())
            print("end1")
            '''

            scene = bge.logic.getCurrentScene()
            lamp = scene.objects['Lamp']
            lamp['SL'] = 0.00001
            #SLOW =0.01

            #camera.localOrientation = (math.radians(66),0-ori,math.radians(70)) # set camera orientation
            #print(ori.z)
            #camera.applyRotation([math.radians(58.0),0,math.radians(52.0+ori.z)],True)
            #camera.applyRotation([0,0,-ori.y],True)
            #fg = ori.z*(math.pow(obj.position[0],2)+math.pow(obj.position[1],2))*math.pi
            #camera.localOrientation = (math.radians(58.0),0,-fg) # set camera orientation
            #print(camera.localOrientation.to_euler())

def reset_pos():


        cont = bge.logic.getCurrentController()
        own = cont.owner
        if 'init' not in own: # Will only run once, or when the var gets removed
            pass
        else:
            del own['init']

        scene = bge.logic.getCurrentScene()
        camera = scene.active_camera
        lamp = scene.objects['Lamp']
        lamp['SL'] = 1
        camera.removeParent()
        #camera.position = (70,-55,47) # set camera position
        camera.worldPosition = [70,-55,47] # set camera position
        camera.localOrientation = [math.radians(58),0,math.radians(52)] # set camera orientation
        '''
        print("st2")
        print(camera.position)
        print(camera.worldOrientation.to_euler())
        print(camera.worldOrientation.to_euler())
        print("end2")
        '''

def main():

    #global TEST
    #print(TEST)
    setup()

    scene = bge.logic.getCurrentScene()
    lamp = scene.objects['Lamp']
    SLOW = lamp['SL']


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
            x.applyRotation([0.0,orbitalVelP[idP]*0.001*SLOW,0.0],True)
            #x.applyRotation([0.0,0.0, orbitalVelP[idP]*0.001],True)
            idP += 1
            #if 'pivot' not in x.name:
            #    oblP.append(x)
        elif 'S_' in x.name:
        #print(x.name)
            x.applyRotation([0.0,orbitalVelS[idS]*0.001*SLOW,0.0],True)
            #x.applyRotation([0.0,0,0, orbitalVelS[idS]*0.001],True)
            idS += 1
        #print(orbitalVel[idx])
    #time.sleep(0.1)

    for ob in scene.objects:
        nn = ob.name
        if 'P_' in nn and 'pivot' not in nn and not 'empty' in nn:
            ob.applyRotation([0.0,ORBITAL_PERIOD[idS]*SLOW,0.0],True)
            oblP.append(ob)
    #for x in oblP:
    #    x.applyRotation([0.0,ORBITAL_PERIOD[idS],0.0],True)


    keyboard= bge.logic.keyboard
    if bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.UPARROWKEY]:
        #bge.render.makeScreenshot('/mnt/NewVolume/studia/s10/TESP/blend/Screenshot#.jpg')
        objL = []
        dataL = []
        xy = []

        cont = bge.logic.getCurrentController()
        own = cont.owner

        if 'init' not in own: # Will only run once, or when the var gets removed
            own['init'] = True
            follow(oblP[6])

    elif bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.DOWNARROWKEY]:
        reset_pos()
        send_to('resetto')
    elif bge.logic.KX_INPUT_ACTIVE == keyboard.events[bge.events.LEFTARROWKEY]:
        bge.render.makeScreenshot('/mnt/NewVolume/studia/s10/TESP/blend/Screenshot#.jpg')
        dataL = []
        for ob in scene.objects:
            nn = ob.name
            if 'P_' in nn and 'pivot' not in nn and not 'empty' in nn:
                #print(ob.worldPosition)
                #print(ob.name)
                #objL.append(ob)
                dataL.append([ob.name[2:], ob.worldPosition[0], ob.worldPosition[1], ob['Diam']])#radius
            #elif 'S_' in nn and 'pivot' not in nn:
        print(dataL)
    else:
        hit = mouse_hit_ray(1,1)
        if hit[0] is None:
            pass
        else:
            print(hit)
        pass
        #reset_pos()
if __name__ == "__main__":
    main()
