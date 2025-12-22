########### FLORIAN KERBRAT MP  ##########

##Imports
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


###♦ POINTS ####
################

##bordel








#### Function ####
##################

def hauteur(a,b,pts_c):

    """calcul 3 nouveau point de controle pour le parametre b,
        puis calcul a partir de ces 3 new points de controle, les coordonées du point A de parametre a
    pm1=coord_pc(a,pts_c[0])
    pm2=coord_pc(a,pts_c[1])
    pm3=coord_pc(a,pts_c[2]) """
    L=[[]]*len(pts_c)
    for i in range(0,len(pts_c)):
        L[i]=coord_pc(a,pts_c[i])

    pf=coord_pc(b,L)

    return (pf[0],pf[1],pf[2])

def Cp(deg):
    """calcul la liste des coeff de pascal pour un polynome de degree n"""
    pa=[1.0]
    for j in range(deg):
        na = pa + [1.0]
        for i in range(0,len(pa)-1):
            na[i+1]=pa[i]+pa[i+1]
        pa=na
    return pa



def coord_pc(t,liste_pc):
    """ calcul les coordonnée d'un point B POUR le paramètre t, sur une courbe de bezier definie par liste_pc
    """
    liste_pascal=Cp(len(liste_pc)-1)
    cc =np.array([0.0, 0.0, 0.0], dtype=float)
    for k in range(len(liste_pc)):
        cc = cc + ((1-t)**(len(liste_pc)-1-k))*(t**(k))*liste_pascal[k]*liste_pc[k]
    return cc



def defZ(Z,X,Y,pts_c,X0,Y0):
    """
    calcul la hauteur z ( la cote ) pour chacun des points de la grille et modifie les coordonnée dans la grille Z
    """
    for l in range(0,50):
        for k in range(0,50):
# Forcer la conversion en float lors de l'assignation
            h = hauteur(X0[l][k],Y0[l][k],pts_c)
            X[l][k] = float(h[0])
            Y[l][k] = float(h[1])
            Z[l][k] = float(h[2])


    return (X,Y,Z)




####     D I S P L A Y   #####
##############################


##préliminaires

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Surface plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

##Tracage d'un carreau de bézier


def carreau(pts_c):

    pts_c = pts_c.astype(float)

    #emplacement de la surface
    tx_min=np.min(pts_c[:,:,0])
    ty_min=np.min(pts_c[:,:,1])

    tx_max=np.max(pts_c[:,:,0])
    ty_max=np.max(pts_c[:,:,1])


    x = np.linspace(tx_min, tx_max, 50)
    y = np.linspace(ty_min, ty_max, 50)
    X, Y = np.meshgrid(x, y)


    "util pour toujours avoir un parametre dans [0,1]*[0,1] et non pas les coords x y si on deplace le carreau"
    x0 = np.linspace(0,1, 50)
    y0 = np.linspace(0,1, 50)
    X0, Y0 = np.meshgrid(x0, y0)

    XBase= np.zeros((50,50))
    YBase= np.zeros((50,50))
    ZBase= np.zeros((50,50))

    XP,YP,ZP=defZ(ZBase,XBase,YBase,pts_c,X0,Y0)

    return(XP,YP,ZP)




def tracage(X,Y,Z,pts_c):

    ax.plot_surface(X,Y,Z, cmap='winter',alpha=1, edgecolor='none', linewidth=0.025, antialiased=False, shade=True)

    #for i in range(len(pts_c)):
    #    for j in range(len(pts_c[i])):
    #        ax.scatter(pts_c[i][j][0], pts_c[i][j][1], pts_c[i][j][2])
    return()


##Translation


def translation_ex(pts_c,x):
    pts_c[:,:,0] = pts_c[:,:,0].astype(float) + x
    return()

def translation_ey(pts_c,y):
    pts_c[:,:,1] = pts_c[:,:,1].astype(float) + y
    return()

def translation_ez(pts_c,z):
    pts_c[:,:,2] = pts_c[:,:,2].astype(float) + z
    return()


##Rotate


def rotation_X_Z(X,Y,Z,pts_c):
    """
    plan Z,Y de hauteur X
    """
    ax.plot_surface(Z,Y,X,cmap='viridis', edgecolor='none')
    return()


def rotation_Y_Z(X,Y,Z,pts_c):
    """
    plan Z X de hauteur Y
    """
    ax.plot_surface(X,Z,Y,cmap='winter',alpha=1, edgecolor='none', linewidth=0.025, antialiased=False, shade=True)

    return()

def rotation_Y_Z_miror(X,Y,Z,pts_c):
    """
    plan Z X de hauteur Y
    """
    ax.plot_surface(-X, -Z, Y, cmap='winter',alpha=1, edgecolor='none', linewidth=0.025, antialiased=False, shade=True)

    return()


def rotation_Y_Z_blanc(X,Y,Z,pts_c):
    """
    plan Z X de hauteur Y
    """
    ax.plot_surface(X,Z,Y,color=(0.88, 0.88, 0.88),alpha=1, edgecolor='none', linewidth=0.025, antialiased=False, shade=True)

    return()




##Finalité


##Debut du nuage

def afinale(pts):

    X,Y,Z = carreau(pts)
    tracage(X,Y,Z,pts)

    return()



def cube(l):

    cbf0= [[l[4],l[5]],
            [l[0],l[1]]]

    cbf1= [[l[5],l[6]],
            [l[1],l[2]]]

    cbf2= [[l[6],l[7]],
            [l[2],l[3]]]

    cbf3= [[l[7],l[4]],
            [l[3],l[0]]]

    cbf4= [[l[7],l[6]],
            [l[4],l[5]]]

    cbf5= [[l[3],l[2]],
            [l[0],l[4]]]

    face_cube = [cbf0,cbf1,cbf2,cbf3,cbf4,cbf5]

    trace_list(face_cube)

    return()


def tout_les_cubes_longeur(lptscube):
    cube(lptscube)
    li = lptscube

    for i in range(0,60,6):
        for j in range(0,len(lptscube)):

            li[j] += np.array([6,0,0])

        cube(li)

    return()

def tout_les_cubes_largeur(lptscube):

    li = lptscube

    for i in range(0,24,9):
        for j in range(0,len(lptscube)):

            li[j] += np.array([0,0,-7])

        cube(li)

    return()


def trace_list(f):
    for i in range(0,len(f)):
        fi = np.array(f[i])
        X, Y, Z = carreau(fi)
        rotation_Y_Z(X,Y,Z, fi)

        trace_list_miror(fi)
    return ()

def trace_list_blanc(f):
    """la seul différence c'est pour changer la couleur en blanc"""
    for i in range(0,len(f)):
        fi = np.array(f[i])
        fi = fi[:, ::-1]
        X, Y, Z = carreau(fi)
        rotation_Y_Z_blanc(X,Y,Z, fi)

        trace_list_miror_blanc(fi)
    return ()



##Meilleur symetrie



def symetrie_centrale(pts_c, centre_x, centre_z):
    """
    Effectue une symétrie centrale par rapport au point (centre_x, centre_z)
    """
    pts_sym = pts_c.astype(float).copy()

    pts_sym[:,:,0] = 2 * centre_x - pts_c[:,:,0]  # X
    pts_sym[:,:,2] = 2 * centre_z - pts_c[:,:,2]  # Z

    return pts_sym


def trace_list_miror(fi):
    centre_x = 31.5
    centre_z = -10.5

    fi_sym = symetrie_centrale(fi, centre_x,centre_z)

    X, Y, Z = carreau(fi_sym)
    rotation_Y_Z(X, Y, Z, fi_sym)

    return()




def trace_list_miror_blanc(fi):
    centre_x = 31.5
    centre_z = -10.5

    fi_sym = symetrie_centrale(fi, centre_x,centre_z)

    X, Y, Z = carreau(fi_sym)
    rotation_Y_Z_blanc(X, Y, Z, fi_sym)

    return()






######################################
################# Le nuage ###########
######################################

##Vue de profil (grande face)


p00=np.array([1.5,17.0,0])
p01=np.array([3,17,0])
p02=np.array([20.25,17,0])
p03=np.array([21.75,17,0])
p04=np.array([41.25,17,0])
p05=np.array([42.75,17,0])
p06=np.array([60,17,0])
p07=np.array([61.5,17,0])



p10=np.array([1.5,16.5,0])
p11=np.array([3,16.5,0])
p12=np.array([11,16,0])
p13=np.array([12,16,0])
p1b=np.array([20.25,16.5,0])
p1b2=np.array([21.75,16.5,0])
p14=np.array([30.5,16,0])
p15=np.array([31.5,16,0])
p16=np.array([41.25,16.5,0])
p17=np.array([42.75,16.5,0])
p18=np.array([52,16,0])
p19=np.array([53,16,0])
p110=np.array([60,16.5,0])
p111=np.array([61.5,16.5,0])


p20=np.array([11,14,0])
p21=np.array([12,14,0])
p22=np.array([30.5,14,0])
p23=np.array([31.5,14,0])
p24=np.array([52,14,0])
p25=np.array([53,14,0])


p30=np.array([1.5,10.5,0])
p31=np.array([3,10.5,0])
p32=np.array([21,11,0])
p33=np.array([22,11,0])
p34=np.array([41,11,0])
p35=np.array([42,11,0])
p36=np.array([61,10.5,0])
p37=np.array([62.5,10.5,0])



p40=np.array([1.5,9.75,0])
p41=np.array([3,9.75,0])
p42=np.array([21,9,0])
p43=np.array([22,9,0])
p44=np.array([41,9,0])
p45=np.array([42,9,0])
p46=np.array([61,9.75,0])
p47=np.array([62.5,9.75,0])





p50=np.array([11,8,0])
p51=np.array([12,8,0])
p52=np.array([30.5,8,0])
p53=np.array([31.5,8,0])
p54=np.array([52,8,0])
p55=np.array([53,8,0])


p60=np.array([11,6,0])
p61=np.array([12,6,0])
p62=np.array([21,5,0])
p63=np.array([22,5,0])
p64=np.array([30.5,6,0])
p65=np.array([31.5,6,0])
p66=np.array([41,5,0])
p67=np.array([42,5,0])
p68=np.array([52,6,0])
p69=np.array([53,6,0])


p70=np.array([1.5,3.5,0])
p71=np.array([3,3.5,0])
p72=np.array([60,3.5,0])
p73=np.array([61.5,3.5,0])


p80=np.array([1.5,3,0])
p81=np.array([3,3,0])
p82=np.array([21,3,0])
p83=np.array([22,3,0])
p84=np.array([41,3,0])
p85=np.array([42,3,0])
p86=np.array([60,3,0])
p87=np.array([61,3,0])




""" figure 1 """
pf10=(0,13,1)
pf12=(11,15,0)
pf11=(2.5,13,5)


f1=[[p10,p11,p12],
    [pf10,pf11,pf12],
    [p30,p31,p20]]




""" figure 2 """

pf20=[3,10.125,0]
pf21=[11,11,4]
pf22=[12,11,4]
pf23=[21,10,0]



f2=[[p31,p20,p21,p32],
    [pf20,pf21,pf22,pf23],
    [p41,p50,p51,p42]]


""" figure 3 """

pf30=(p13+p21)/2
pf31=(p1b+p32)/2 + np.array([0,0,4])
pf32=(p1b2+p33)/2 + np.array([0,0,4])
pf33=(p14+p22)/2


f3=[[p13,p1b,p1b2,p14],
    [pf30,pf31,pf32,pf33],
    [p21,p32,p33,p22]]

""" figure 4 """


pf40=(p33+p43)/2
pf41=(p22+p52)/2 + np.array([0,0,4])
pf42=(p23+p53)/2 + np.array([0,0,4])
pf43=(p34+p44)/2


f4=[[p33,p22,p23,p34],
    [pf40,pf41,pf42,pf43],
    [p43,p52,p53,p44]]



""" figure 5 """

pf50=(p15+p23)/2
pf51=(p16+p34)/2 + np.array([0,0,4])
pf52=(p17+p35)/2 + np.array([0,0,4])
pf53=(p18+p24)/2



f5=[[p15,p16,p17,p18],
    [pf50,pf51,pf52,pf53],
    [p23,p34,p35,p24]]


""" figure 6 """

pf60=(p35+p45)/2
pf61=(p24+p54)/2 + np.array([0,0,4])
pf62=(p25+p55)/2 + np.array([0,0,4])
pf63=(p36+p46)/2


f6=[[p35,p24,p25,p36],
    [pf60,pf61,pf62,pf63],
    [p45,p54,p55,p46]]


""" figure 7 """

pf70=(p19+p25)/2
pf71=(p110+p36)/2 + np.array([0,0,4])
pf72=(p111+p37)/2 + np.array([1.5,0,2])



f7=[[p19,p110,p111],
    [pf70,pf71,pf72],
    [p25,p36,p37]]

""" figure 8"""


pf80=(p40+p70)/2 + np.array([-1.5,0,1])
pf81=(p41+p71)/2 + np.array([0,0,4])
pf82 = (p50+p60)/2

f8 = [[p40,p41,p50],
        [pf80,pf81,pf82],
        [p70,p71,p60]]

""" figure 9"""


pf90=(p51+p61)/2
pf91=(p42+p62)/2 + np.array([0,0,4])
pf92 = (p43+p63)/2 + np.array([0,0,4])
pf93 = (p52+p64)/2


f9 = [[p51,p42,p43,p52],
        [pf90,pf91,pf92,pf93],
        [p61,p62,p63,p64]]


""" figure 10"""

pf104 = np.array([2,3,0])
pf105 = np.array([2.5,3,0])

pf100=(p71+p81)/2
pf101=(p60+pf104)/2 + np.array([0,0,4])
pf102 = (p61+pf105)/2 + np.array([0,0,4])
pf103 = (p62+p82)/2



f10 = [[p71,p60,p61,p62],
        [pf100,pf101,pf102,pf103],
        [p81,pf104,pf105,p82]]


""" figure 11"""


pf110=(p53+p65)/2
pf111=(p44+p66)/2 + np.array([0,0,4])
pf112 = (p45+p67)/2 + np.array([0,0,4])
pf113 = (p54+p68)/2



f11 = [[p53,p44,p45,p54],
        [pf110,pf111,pf112,pf113],
        [p65,p66,p67,p68]]

""" figure 12"""

pf124 = np.array([28.3,3,0])
pf125 = np.array([34.6,3,0])

pf120=(p63+p83)/2
pf121=(p64+pf124)/2 + np.array([0,0,4])
pf122 = (p65+pf125)/2 + np.array([0,0,4])
pf123 = (p66+p84)/2



f12 = [[p63,p64,p65,p66],
        [pf120,pf121,pf122,pf123],
        [p83,pf124,pf125,p84]]


""" figure 13"""

pf134 = np.array([48,3,0])
pf135 = np.array([54,3,0])

pf130=(p67+p85)/2
pf131=(p68+pf134)/2 + np.array([0,0,4])
pf132 = (p69+pf135)/2 + np.array([0,0,4])
pf133 = (p72+p86)/2



f13 = [[p67,p68,p69,p72],
        [pf130,pf131,pf132,pf133],
        [p85,pf134,pf135,p86]]


""" figure 14"""

pf140=(p55+p69)/2
pf141=(p46+p72)/2 + np.array([0,0,4])
pf142 = (p47+p73)/2 + np.array([1.5,0,2])



f14 = [[p55,p46,p47],
        [pf140,pf141,pf142],
        [p69,p72,p73]]



""" figure top 1 """

pt10 = np.array([8.75,17,0])
pt11 = np.array([14.5,17,0])


t1 = [[p01,pt10,pt11,p02],
        [p11,p12,p13,p1b]]



""" figure top 2 """

pt20 = np.array([28.25,17,0])
pt21 = np.array([34.75,17,0])


t2 = [[p03,pt20,pt21,p04],
        [p1b2,p14,p15,p16]]

""" figure top 3 """



pt30 = np.array([48.5,17,0])
pt31 = np.array([54.25,17,0])


t3 = [[p05,pt30,pt31,p06],
        [p17,p18,p19,p110]]

""" figure jointure j de profil """

j0 = [[p00,p01],
        [p10,p11]]

j1 = [[p02,p03],
        [p1b,p1b2]]

j2 = [[p04,p05],
        [p16,p17]]

j3 = [[p06,p07],
        [p110,p111]]

j4 = [[p12,p13],
        [p20,p21]]

j5 = [[p14,p15],
        [p22,p23]]

j6 = [[p18,p19],
        [p24,p25]]

j7 = [[p30,p31],
        [p40,p41]]

j8 = [[p32,p33],
        [p42,p43]]

j9 = [[p34,p35],
        [p44,p45]]

j10 = [[p36,p37],
        [p46,p47]]

j11 = [[p50,p51],
        [p60,p61]]

j12 = [[p52,p53],
        [p64,p65]]

j13 = [[p54,p55],
        [p68,p69]]

j14 = [[p70,p71],
        [p80,p81]]

j15 = [[p62,p63],
        [p82,p83]]

j16 = [[p66,p67],
        [p84,p85]]

j17 = [[p72,p73],
        [p86,p87]]


jtab =[j0,j1,j2,j3,j4,j5,j6,j7,j8,j9,j10,j11,j12,j13,j14,j15,j16,j17]
ftab = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
ttab = [t1,t2,t3]







##Vue de face (petite face)

q00=np.array([1.5,17,-21])
q01=np.array([1.5,17,-19.5])
q02=np.array([1.5,17,-1.5])
q03=np.array([1.5,17,0])


#q10=np.array([1.5,16,-21])
#q11=np.array([1.5,16,-19.5])
#q12=np.array([1.5,16,-1.5])
#q13=np.array([1.5,16,0])
q10=np.array([1.5,16.5,-21])
q11=np.array([1.5,16.5,-19.5])
q12=np.array([1.5,16.5,-1.5])
q13=np.array([1.5,16.5,0])


q20=np.array([1.5,13.6,-11])
q21=np.array([1.5,13.6,-10])
q22=np.array([1.5,13.3,-11])
q23=np.array([1.5,13.3,-10])

q30=np.array([1.5,10.5,-21])
q31=np.array([1.5,10.5,-19.5])
q32=np.array([1.5,10.5,-1.5])
q33=np.array([1.5,10.5,0])

q40=np.array([1.5,9.75,-21])
q41=np.array([1.5,9.75,-19.5])
q42=np.array([1.5,9.75,-1.5])
q43=np.array([1.5,9.75,0])

q50=np.array([1.5,6.6,-11])
q51=np.array([1.5,6.6,-10])
q52=np.array([1.5,6.3,-11])
q53=np.array([1.5,6.3,-10])

q60=np.array([1.5,3.5,-21])
q61=np.array([1.5,3.5,-19.5])
q62=np.array([1.5,3.5,-1.5])
q63=np.array([1.5,3.5,0])

q71=np.array([1.5,3,-21])
q72=np.array([1.5,3,-19.5])
q75=np.array([1.5,3,-12])
q76=np.array([1.5,3,-9])
q79=np.array([1.5,3,-1.5])
q710=np.array([1.5,3,0])


""" figure face 1 """

qp10 =np.array([0,13,1])
qp11=(q12+q32)/2 + np.array([-4,0,0])
qp12 = (q21+q23)/2


fp1 = [[q21,q12,q13],
        [qp12,qp11,qp10],
        [q23,q32,q33]]





""" figure face 2 """

qp20 =np.array([1.5,17,-10.5])
qp21 =np.array([1.5,17,-16.5])

qp23=(q11+q01)/2
qp26=(q02+q12)/2

qp24 =np.array([-5.5,15.5,-12.07])
qp25 =np.array([-5.5,15.5,-8.91])


fp2 = [[q01,qp20,qp21,q02],
        [qp23,qp24,qp25,qp26],
        [q11,q20,q21,q12]]


""" figure face 3 """


qp30= (q31+q41)/2
qp33= (q32+q42)/2
qp31= (q22+q50)/2 + np.array([-4,0,0])
qp32= (q23+q51)/2 + np.array([-4,0,0])

fp3= [[q31,q22, q23, q32],
        [qp30,qp31,qp32,qp33],
        [q41,q50,q51,q42]]

""" figure face 4 """


qp40= (q51+q53)/2
qp41= (q42+q62)/2 + np.array([-4,0,0])
qp42 = pf80

fp4= [[q51,q42,q43],
        [qp40,qp41,qp42],
        [q53,q62,q63]]

""" figure face 5 """


qp54= np.array([1.5,3,-6])
qp55= np.array([1.5,3,-15])

qp50= (q61+q72)/2
qp53= (q62+q79)/2

qp51= np.array([-5.5,4.3,-12.8])
qp52= np.array([-5.5,4.3,-8.5])


fp5= [[q61,q52, q53, q62],
        [qp50,qp51,qp52,qp53],
        [q72,qp54,qp55,q79]]


""" figure face 6 """

pf10= np.array([0,13,1])

qp60= np.array([0,13,-22])

qp61= (q11+q31)/2 + np.array([-5.5,0,0])
qp62= (q20+q22)/2

fp6 = [[q10,q11, q20],
        [qp60,qp61,qp62],
        [q30,q31,q22]]

""" figure face 7 """

pf10= np.array([0,13,1])

qp70= np.array([0,6.5,-22])
qp71= (q41+q61)/2 + np.array([-5.5,0,0])
qp72= (q50+q52)/2

fp7 = [[q40,q41, q50],
        [qp70,qp71,qp72],
        [q60,q61,q52]]



""" figue jointure j """

jff0 = [[q00,q01],
        [q10,q11]]

jff1 = [[q02,q03],
        [q12,q13]]


jff2 = [[q20,q21],
        [q22,q23]]

jff3 = [[q30,q31],
        [q40,q41]]


jff4 = [[q32,q33],
        [q42,q43]]

jff5 = [[q50,q51],
        [q52,q53]]


jff6 = [[q60,q61],
        [q71,q72]]

jff7 = [[q62,q63],
        [q79,q710]]



fjptab = [jff0, jff1, jff2, jff3, jff4, jff5, jff6, jff7, ]

fptab = [fp1,fp2,fp3,fp4,fp5,fp6,fp7]



##Le socle

#premire rangée

ps00=np.array([0,0,1.5])
ps01=np.array([3,0,1.5])
ps02=np.array([3,0,-1.5])
ps03=np.array([0,0,-1.5])

ps10=np.array([0,3,1.5])
ps11=np.array([3,3,1.5])
ps12=np.array([3,3,-1.5])
ps13=np.array([0,3,-1.5])

#2eme rangée

ps200=np.array([0,0,-19.5])
ps201=np.array([3,0,-19.5])
ps202=np.array([3,0,-22.5])
ps203=np.array([0,0,-22.5])

ps210=np.array([0,3,-19.5])
ps211=np.array([3,3,-19.5])
ps212=np.array([3,3,-22.5])
ps213=np.array([0,3,-22.5])

#3eme rangé de cubes pour la largeur opposée

ps300=np.array([60,0,1.5])
ps301=np.array([63,0,1.5])
ps302=np.array([63,0,-1.5])
ps303=np.array([60,0,-1.5])

ps310=np.array([60,3,1.5])
ps311=np.array([63,3,1.5])
ps312=np.array([63,3,-1.5])
ps313=np.array([60,3,-1.5])



lptscube = np.array([ps00,ps01,ps02,ps03,ps10,ps11,ps12,ps13]) #premier cube de la longeur 1
lptscube2 = np.array([ps200,ps201,ps202,ps203,ps210,ps211,ps212,ps213]) #premier cube de la longeur 2
lptscube3 = np.array([ps300,ps301,ps302,ps303,ps310,ps311,ps312,ps313])




##affichage de toutes les figures

#pour avoir une resolution fixe. Un rectangle
ax.scatter(30,15,10)
ax.scatter(30,-45,10)
ax.scatter(0,0,15)
ax.scatter(0,0,-15)

#ax.scatter(0,0,22.5)
#ax.scatter(0,0,-7.5)


#Tracé de la grande face
trace_list(ftab) # la grosse partie de la grande face
trace_list_blanc(jtab) # les jointure de la grande face
trace_list(ttab) # le sous toit de la grande face


#Tracé de la petite face
trace_list(fptab) #la petite face
trace_list_blanc(fjptab) #jointure de la petite face


# Tracé des cubes
tout_les_cubes_longeur(lptscube)
tout_les_cubes_largeur(lptscube)



##Final

ax.view_init(elev=0, azim=0) # Vu de la petite face
#ax.view_init(elev=90, azim=0) #Vu du dessus

#ax.view_init(0, 90) # vu de la grande face


plt.show()


