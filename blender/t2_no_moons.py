import bpy
import bmesh
import mathutils

import threading
import math
import time

SCALE_CENTER = .000003
SCALE_DIAM=.0001
SCALE_DIST = .00000001
SUN = ['Sun',1390000,1.989*math.pow(10,30),0,0, 0]
PLANETS=['Mercury','Venus','Earth', 'Mars', 'Jupiter','Saturn','Uranus', 'Neptune']
DIAMETERS = [4879,12104,12756,6792,142984,120536,51118,49528]
DIAMETERS[4:] = [x*0.12  for x in DIAMETERS[4:]]

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

        o = bpy.context.active_object
        o.select = True
        new_mat = bpy.data.materials.new(name="a")
        new_mat.diffuse_color = (0.8,0.8,0.0)
        o.data.materials.append(new_mat)

def create_system(center, orbiters):
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
    o.parent = center.blenderObj # parent pivot to center
    x.blenderObj.parent = o #parent object to pivot
    empt.append(o)
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
    text = bpy.data.texts.load('/mnt/NewVolume/studia/s10/TESP/blend/bgee.py')

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

    camera = bpyscene.objects["Camera"]
    camera.rotation_euler = (0, 0.0, 0)
    camera.location.x = 0.0
    camera.location.y = 0.0
    camera.location.z = 115.0
    bpy.data.cameras['Camera'].clip_end =500


def main():
    setupW()
    bpy.data.objects.remove(bpy.data.objects['Cube'],True)
    star, planetslist, empt = create_system(SUN, ALL_DATA)
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
