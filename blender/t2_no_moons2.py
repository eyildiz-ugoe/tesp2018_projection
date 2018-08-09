import bpy
import bmesh
import mathutils

import sys, os
import threading
import math
import time

#TEXTURES_PATH = '/mnt/NewVolume/studia/s10/TESP/tesp2018_projection/textures/'
FILE_PATH = bpy.context.space_data.text.filepath[:-15]
TEXTURES_PATH =  FILE_PATH+'/textures/'
#TEXTURES_PATH =os.path.dirname(os.path.abspath(__file__))+'/textures/'

SCALE_CENTER = .000003
SCALE_DIAM=.0001
SCALE_DIST = .00000001
SUN = ['S_Sun',1390000,1.989*math.pow(10,30),0,0, 0]
PLANETS=['Mercury','Venus','Earth', 'Mars', 'Jupiter','Saturn','Uranus', 'Neptune']
DIAMETERS = [4879,12104,12756,6792,142984,120536,51118,49528]
DIAMETERS[4:6] = [x*0.12  for x in DIAMETERS[4:6]]
DIAMETERS[6:] = [x*0.2  for x in DIAMETERS[6:]]

#diameters = []
MASSES = [0.330,4.87,5.97,0.642,1898,568,86.8,102]
MASSES = [x * math.pow(10,24) for x in MASSES]
#distance = [57.9,108.2,149.6,227.9,778.6,1433.5,2872.5,4495.1]
DISTANCES = [500 * x for x in range(1,len(PLANETS)+1)]
DISTANCES[0] = 500
DISTANCES = [x * math.pow(10,6) for x in DISTANCES]
ORBITAL_VEL = [47.4,35.0,29.8,24.1,13.1,9.7,6.8,5.4]
orbitalVelP = ORBITAL_VEL[::-1]
orbitalVelP = [x*0.5  for x in orbitalVelP]
orbitalVelS = [10.8791667, 1]
ORBITAL_PERIOD = [88.0,224.7,365.2,687.0,4331,10747,30589,59800]
#orbitalPeriod = [88.0,224.7,365.2,687,433,1074,3058,5980]
# = [int(x) for x in orbitalPeriod]
ORBITAL_PERIOD[4:] = [x*0.01  for x in ORBITAL_PERIOD[4:]]

ALL_DATA = []#[PLANETS, DIAMETERS, MASSES, DISTANCES, ORBITAL_VEL, ORBITAL_PERIOD]
for x in range(len(PLANETS)):
    #ALL_DATA.append(['P_'+PLANETS[x], DIAMETERS[x], MASSES[x], DISTANCES[x], ORBITAL_VEL[x], ORBITAL_PERIOD[x]])
    ALL_DATA.append(['P_'+PLANETS[x], DIAMETERS[x], MASSES[x], DISTANCES[x], ORBITAL_VEL[x], ORBITAL_PERIOD[x]])
m1 = [['Moon', 3476, 7.34767309* math.pow(10,22), 384400, 1,	27.322]]
m1 = [['Moon', 3476, 7.34767309* math.pow(10,22), 130000000, 1,	27.322]]
m5 = [['Ganymede',5262.4, 14819000*math.pow(10,16), 1070400 , 10.8791667, 7.1546]]
m5 = [['Ganymede',5262.4, 14819000*math.pow(10,16), 160000000 , 10.8791667, 7.1546]]
SATELLITES = [0,0,m1,0,m5,0,0,0]
SATELLITES = [0,0,0,0,0,0,0,0]
moonsD = [0,0, 400,0,0, 800]
moonsDis = [50, 100]

class Center(object):
    def __init__(self,name,diameter, mass, distance, orbitalVel, orbitalPeriod,dRef):#RS,theta0,radius):
        self.name = name
        self.diameter = diameter*SCALE_CENTER
        self.mass = mass
        self.distance = distance*SCALE_DIST
        self.dRef = dRef
        self.orbitalVel = orbitalVel
        self.orbitalPeriod = orbitalPeriod
        self.blenderObj = None
        self.pos = 0

    def create3d(self):
        bpyscene = bpy.context.scene
        # Create an empty mesh and the object.
        mesh = bpy.data.meshes.new(self.name)
        basic_sphere = bpy.data.objects.new(self.name, mesh)

        #basic_sphere['Diam'] = self.diameter # set property to be accessible from other script
        self.blenderObj = basic_sphere



        # Add the object into the scene.
        bpyscene.objects.link(basic_sphere)
        bpyscene.objects.active = basic_sphere
        basic_sphere.select = True


        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        bpy.ops.object.game_property_new()#NOTE
        prop = basic_sphere.game.properties[-1]
        prop.name = 'Diam'
        prop = basic_sphere.game.properties[-1]
        prop.value = self.diameter



        # Construct the bmesh cube and assign it to the blender mesh.
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=self.diameter)
        bm.to_mesh(mesh)
        bm.free()

        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.ops.object.shade_smooth()

class Orbiter(Center):
    def __init__(self,name,diameter, mass, distance,orbitalVel, orbitalPeriod, dRef):#RS,theta0,radius):
        super(Orbiter, self).__init__(name,diameter,mass,distance, orbitalVel, orbitalPeriod,dRef)
        self.diameter = diameter*SCALE_DIAM
        #self.pos = 0

    def place(self):
        a = self.distance + self.dRef
        self.blenderObj.location.x += a
        self.pos = a
        #orbit
        bpy.ops.mesh.primitive_circle_add(vertices = 128, radius = a)



def create_system(center, orbiters):
    #universe sphere
    bpyscene = bpy.context.scene
    # Create an empty mesh and the object.
    mesh = bpy.data.meshes.new('uni')
    basic_sphere = bpy.data.objects.new('uni', mesh)
    bpyscene.objects.link(basic_sphere)
    bpyscene.objects.active = basic_sphere
    basic_sphere.select = True
    # Construct the bmesh cube and assign it to the blender mesh.
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, diameter=300)
    bmesh.ops.reverse_faces(bm, faces = bm.faces) #flip normals to have texture on the inside
    bm.to_mesh(mesh)
    bm.free()
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.shade_smooth()
    realpath = TEXTURES_PATH+'Stars.jpg'

    try:
        img = bpy.data.images.load(realpath)
    except:
        raise NameError("Cannot load image %s" % realpath)
    # Create image texture from image
    cTex = bpy.data.textures.new('Stars' + "_texture", type = 'IMAGE')
    cTex.image = img

    mat = bpy.data.materials.new(name='Stars' + "_material")
    mat.emit = 2.0
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.0
    mat.game_settings.use_backface_culling = False
    mtex = mat.texture_slots.add()
    mtex.texture = cTex
    mtex.texture_coords = 'OBJECT'
    #mtex.texture_coords = 'ORCO'
    #mtex.object = dummy_object
    mtex.use_map_color_diffuse = True
    mtex.use_map_color_emission = True
    #mtex.emission_color_factor = 0.5
    #mtex.use_map_density = True
    mtex.mapping = 'SPHERE'#'FLAT'


    basic_sphere.data.materials.append(mat)
    basic_sphere.rotation_euler.x = math.radians(130)
    basic_sphere.rotation_euler.y = math.radians(130)
    basic_sphere.rotation_euler.z = math.radians(130)
















    #if list == None:
    objects_list =[]
    e=0
    star = Center(*center,0)
    star.create3d()
    ref = star.diameter/2
    for x in orbiters:
        objects_list.append(Orbiter(*x,ref))
        objects_list[-1].create3d()
        objects_list[-1].place()
        e = create_rotation(objects_list[-1],star)

    #o = bpy.data.objects.new( "Center", None )
    #bpy.context.scene.objects.link( o )

    return star, objects_list, e


def create_moons(center, orbiters):
    objects_list =[]
    ref = center.diameter/2
    for x in orbiters:
        x[0] = 'S_' + x[0]
        objects_list.append(Orbiter(*x,ref))
        objects_list[-1].create3d()
        objects_list[-1].place()
        e = create_rotation(objects_list[-1],center)
    #o = bpy.data.objects.new( "Planet center", None )
    #bpy.context.scene.objects.link( o )
    return objects_list, e



def create_rotation(x, center):
    empt = []
    o = bpy.data.objects.new( "pivot_"+x.name, None )
    bpy.context.scene.objects.link( o )
    o.location.x += center.pos
    o.rotation_mode='XYZ'
    #o.rotation_euler = (math.radians(90), 0, 0)
    o.parent = center.blenderObj # parent pivot to center
    #o.matrix_parent_inverse = center.blenderObj.matrix_world.inverted()
    x.blenderObj.parent = o #parent object to pivot
    empt.append(o)



    #for cameras

    o2 = bpy.data.objects.new( "empty_"+x.name, None )
    bpy.context.scene.objects.link( o2 )
    o2.location.x += x.pos -(x.diameter+3)
    o2.location.y += 0
    o2.location.z += 1
    o2.rotation_mode='XYZ'
    #o2.rotation_euler.x = math.radians(90)
    o2.rotation_euler.y = math.radians(-90)
    #o2.rotation_euler.z= math.radians(0)
    #o.rotation_euler = (math.radians(90), 0, 0)
    #o2.parent = o # parent pivot to center
    #o.matrix_parent_inverse = center.blenderObj.matrix_world.inverted()
    o2.parent = o #parent object to pivot
    return empt


def rotate(o, speeds):
    scene = bpy.data.scenes["Scene"]
    deadzone = 80
    scene.frame_start = 1
    scene.frame_end = max(speeds)
    #y=0
    for idx, x in enumerate(o):
        #for y in range(0, max(speeds),speeds[idx]):
        x.rotation_mode='XYZ'
        x.rotation_euler = (0, 0, 0)
        x.keyframe_insert('rotation_euler', index=2 ,frame=0)

        x.rotation_euler = (0, 0, math.radians(360))
        #x.keyframe_insert('rotation_euler', index=idx ,frame=speeds[idx])
        x.keyframe_insert('rotation_euler', index=2 ,frame=speeds[idx])
        for fc in x.animation_data.action.fcurves:
            fc.extrapolation = 'LINEAR' # Set extrapolation type
            fc.modifiers.new('CYCLES')
            for f in fc.keyframe_points:
                f.interpolation = 'LINEAR'

        #time.sleep(0.1)
    #bpy.ops.render.render(animation=True)

def setupW():
    bpy.context.scene.render.engine = 'BLENDER_GAME'

    text = bpy.data.texts.load(FILE_PATH+'bgee.py')

    for area in bpy.context.screen.areas:
        if area.type == 'TEXT_EDITOR':
            area.spaces[0].text = text # make loaded text file visible

            ctx = bpy.context.copy()
            ctx['edit_text'] = text # specify the text datablock to execute
            ctx['area'] = area # not actually needed...
            ctx['region'] = area.regions[-1] # ... just be nice

            #bpy.ops.text.run_script(ctx)
            break


    bpyscene = bpy.context.scene
    bpyscene.game_settings.material_mode = 'GLSL'
    bpyscene.game_settings.use_glsl_shaders = False

    camera = bpyscene.objects["Camera"]
    camera.rotation_euler = (0, 0.0, 0)
    camera.location.x = 0.0
    camera.location.y = 0.0
    camera.location.z = 173.0
    #bpy.data.cameras['Camera'].clip_end =500
    camera.data.clip_end =600


    bpyscene.objects.active = camera
    camera.select = True
    bpy.ops.object.game_property_new()
    prop = camera.game.properties[-1]
    prop.name = 'SL'
    prop = camera.game.properties[-1]
    prop.value = 1





    Lamp = bpyscene.objects['Lamp']
    Lamp.rotation_euler = (0, 0.0, 0)
    Lamp.location.x = 0.0
    Lamp.location.y = 0.0
    Lamp.location.z = 0.0
    Lamp.data.energy = 10
    Lamp.data.use_specular = False



    #obj = bpy.context.object
    sensors = camera.game.sensors
    controllers = camera.game.controllers
    #actuators = obj.game.actuators
    bpy.ops.logic.sensor_add(type="ALWAYS", object=camera.name)
    bpy.ops.logic.controller_add(type="PYTHON", object=camera.name)
    #bpy.ops.logic.actuator_add(type="ACTION", object=obj.name)
    sensor = sensors[-1]
    controller = controllers[-1]
    #actuator = actuators[-1]
    sensor.link(controller)
    sensor.use_pulse_true_level = True
    controller.text = bpy.data.texts['bgee.py']
    #actuator.link(controller)

    '''
    dummy_object = bpy.data.objects.new( 'DUMMY', None )
    bpy.context.scene.objects.link( dummy_object )
    dummy_object.rotation_mode='XYZ'
    dummy_object.rotation_euler = (89.9, 0, 0)

    return dummy_object
    '''
def make_texture(object,emit):
    x = object.blenderObj
    #bpy.context.scene.objects.active = x
    #ob = bpy.context.active_object
    #mat = bpy.data.materials.new(name=y.name + "_material")
    '''
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 1.0
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    '''

    # Assign it to object
    #if x.data.materials:
        # assign to 1st material slot
    #    x.data.materials[0] = mat
    #else:
        # no slots
    #x.data.materials.append(mat)



    #realpath = os.path.expanduser('/mnt/NewVolume/studia/s10/TESP/tesp2018_projection/'+y.name + '.jpg')
    realpath = TEXTURES_PATH+object.name[2:] + '.jpg'

    try:
        img = bpy.data.images.load(realpath)
    except:
        raise NameError("Cannot load image %s" % realpath)
    # Create image texture from image
    cTex = bpy.data.textures.new(object.name[2:] + "_texture", type = 'IMAGE')
    cTex.image = img

    mat = bpy.data.materials.new(name=object.name[2:] + "_material")
    mat.emit = emit
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.0
    mtex = mat.texture_slots.add()
    mtex.texture = cTex
    mtex.texture_coords = 'OBJECT'
    #mtex.object = dummy_object
    mtex.use_map_color_diffuse = True
    mtex.use_map_color_emission = True
    mtex.emission_color_factor = 0.5
    mtex.use_map_density = True
    mtex.mapping = 'SPHERE'#'FLAT'

    x.data.materials.append(mat)


    #GameLogic.video.source = bge.texture.VideoFFmpeg(movie)

def main():
    setupW()
    bpy.data.objects.remove(bpy.data.objects['Cube'],True)
    star, planetslist, empt = create_system(SUN, ALL_DATA)
    star.blenderObj.rotation_euler = (math.radians(90), 0, 0)
    #bpy.context.scene.objects.active = star.blenderObj
    #star.blenderObj.select = True
    #bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    make_texture(star,emit = 2.0)
    for x in planetslist:
        make_texture(x,emit = 0.0 )
    for idx, x in enumerate(planetslist):
        if SATELLITES[idx] == 0:
            continue
        else:
            #for y in SATELLITES:
            #    y[0] = 'S_' + y[0]
            moonslist, empt2 = create_moons(x, SATELLITES[idx])
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D' or area.type == 'Animation':
            print(area.spaces[0])
            area.spaces[0].show_relationship_lines = False


if __name__ == "__main__":
    main()
