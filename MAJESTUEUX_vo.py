########### FLORIAN KERBRAT MP  ##########

##Imports
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


###♦ POINTS ####
################

##bordel




"""

pts_c_l=[[[0,0,0],[5,0,10],[10,0,0],[14,0,2]],
       [[0,2,2],[5,2,15],[10,2,5],[14,2,2]],
       [[0,6,0],[5,6,7],[10,6,1],[14,6,2]]]

pts_c_l=[[[0,0,0],[0,1/2,1],[0,1,0]],
       [[1/2,0,1],[1/2,1/2,8],[1/2,1,1]],
       [[1,0,0],[1,1/2,1],[1,1,0]]]

pts_c=np.array(pts_c_l)



ps1 = np.array([[5,5,5],[2,6,10],[0,50,0]])
ps1 += 10
ps2 = np.array([[7,7,7],[2,6,10],[5,0,5]])

ps3 = np.array([[1,1,1],[2,6,10],[0,10,0]])




pts_c_l2=[[[15,15,15],[12,16,20],[10,60,10]],
       [[7,7,7],[2,6,10],[5,0,5]],
       [[1,1,1],[2,6,10],[0,10,0]]]



pts_c=np.array(pts_c_l2)"""



##rangé

pts_c_l0=[[[0,0,0],[5,0,10],[10,0,0],[14,0,2]],
       [[0,2,2],[5,2,15],[10,2,5],[14,2,2]],
       [[0,6,0],[5,6,7],[10,6,1],[14,6,2]]]

pts_c_regular=[[[0,0,0],[0,1/2,1],[0,1,0]],
       [[1/2,0,1],[1/2,1/2,8],[1/2,1,1]],
       [[1,0,0],[1,1/2,1],[1,1,0]]]


pts_c_regular_rev=[[[0,0,0],[0,1/2,-1],[0,1,0]],
       [[1/2,0,-1],[1/2,1/2,-8],[1/2,1,-1]],
       [[1,0,0],[1,1/2,-1],[1,1,0]]]

pts_c_regular_transl=[[[1,1,0],[1,3/2,1],[1,2,0]],
       [[3/2,1,1],[3/2,3/2,8],[3/2,2,1]],
       [[2,1,0],[2,3/2,1],[2,2,0]]]

pts_c_regular_transl_rev=[[[1,1,0],[1,3/2,-1],[1,2,0]],
       [[3/2,1,-1],[3/2,3/2,-8],[3/2,2,-1]],
       [[2,1,0],[2,3/2,-1],[2,2,0]]]

pts_cercle=[[[1,0,0],[2,0,0],[3,0,0],[4,0,1],[4,0,2],[4,0,3],[3,0,4],[2,0,4],[1,0,4],[0,0,3],[0,0,2],[0,0,1],[1,0,0]],[[1,10,0],[2,10,0],[3,10,0],[4,10,1],[4,10,2],[4,10,3],[3,10,4],[2,10,4],[1,10,4],[0,10,3],[0,10,2],[0,10,1],[1,10,0]]]



pts_cir=np.array(pts_cercle)
pts_reg=np.array(pts_c_regular)
pts_reg_rev=np.array(pts_c_regular_rev)
pts_reg_tra=np.array(pts_c_regular_transl)
pts_reg_tra_rev=np.array(pts_c_regular_transl_rev)





#### Function ####
##################

def hauteur(a,b,pts_c):

    """calcul 3 nouveau point de controle pour le parametre b,
        puis calcul a partir de ces 3 new points de controle, les coordonées du point A de parametre a
    pm1=coord_pc(a,pts_c[0])
    pm2=coord_pc(a,pts_c[1])
    pm3=coord_pc(a,pts_c[2])

    """

    L=[[]]*len(pts_c)
    for i in range(0,len(pts_c)):
        L[i]=coord_pc(a,pts_c[i])

    pf=coord_pc(b,L)

    return (pf[0],pf[1],pf[2])

def Cp(deg):
    """calcul la liste des coeff de pascal pour un polynome de degree n"""
    pa=[1]
    for j in range(deg):
        na = pa + [1]
        for i in range(0,len(pa)-1):
            na[i+1]=pa[i]+pa[i+1]
        pa=na
    return pa



def coord_pc(t,liste_pc):
    """ calcul les coordonnée d'un point B POUR le paramètre t, sur une courbe de bezier definie par liste_pc
    """

    liste_pascal=Cp(len(liste_pc)-1)

    cc =np.array([0,0,0])

    for k in range(len(liste_pc)):

        cc = cc + ((1-t)**(len(liste_pc)-1-k))*(t**(k))*liste_pascal[k]*liste_pc[k]
    return cc



def defZ(Z,X,Y,pts_c,X0,Y0):
    """
    calcul la hauteur z ( la cote ) pour chacun des points de la grille et modifie les coordonnée dans la grille Z
    """
    for l in range(0,50):
        for k in range(0,50):
            X[l][k],Y[l][k],Z[l][k]= hauteur(X0[l][k],Y0[l][k],pts_c)

            #if k==49 :
            #print("voici Z pour ",l,k,": ",Z[l][k])

                #print("voici Y pour ",l,k,": ",Y[l][k])
                #print("voici z pour ",k,": ",Z[l][k])
            #print(Z[l][k])
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


    ax.plot_surface(X,Y,Z,cmap='viridis', edgecolor='none')

    for i in range(len(pts_c)):
        for j in range(len(pts_c[i])):
            ax.scatter(pts_c[i][j][0], pts_c[i][j][1], pts_c[i][j][2])


    #print("voici les points aux coins",Z[0][0],Z[0][49],Z[49][0],Z[49][49])
    return()


##Translation


def translation_ex(pts_c,x):
    print("teeeeee")
    pts_c[:,:,0]+=x
    return()
def translation_ey(pts_c,y):
    pts_c[:,:,1]+=y
    return()
def translation_ez(pts_c,z):
    pts_c[:,:,2]+=z
    return()


def translation_exr(pts_c,x):
    print("teeeeee")
    pts_c[:,:,0]+=x
    return(pts_c)
def translation_eyr(pts_c,y):
    pts_c[:,:,1]+=y
    return(pts_c)
def translation_ezr(pts_c,z):
    pts_c[:,:,2]+=z
    return(pts_c)




##Rotate


def rotation_X_Z(X,Y,Z,pts_c):
    """
    plan Z,Y de hauteur X
    """
    ax.plot_surface(Z,Y,X,cmap='viridis', edgecolor='none')

    #for i in range(len(pts_c)):
    #    for j in range(len(pts_c[i])):
    #        ax.scatter(pts_c[i][j][1], pts_c[i][j][2], pts_c[i][j][0])


    return()


def rotation_Y_Z(X,Y,Z,pts_c):
    """
    plan Z X de hauteur Y
    """
    ax.plot_surface(X,Z,Y,cmap='viridis', edgecolor='none')

    #for i in range(len(pts_c)):
    #    for j in range(len(pts_c[i])):
    #        ax.scatter(pts_c[i][j][0], pts_c[i][j][2], pts_c[i][j][1])


    return()


def rotation_Y_Z_inv(X,Y,Z,pts_c):
    """
    plan Z X de hauteur Y
    """
    ax.plot_surface(X,-Z,Y,cmap='viridis', edgecolor='none')


    #for i in range(len(pts_c)):
    #    for j in range(len(pts_c[i])):
    #        ax.scatter(pts_c[i][j][0], pts_c[i][j][2], pts_c[i][j][1])


    return()







##Finalité

##carré : ok!

#cercle"""
"""
X0,Y0,Z0=carreau(pts_cir)
tracage(X0,Y0,Z0,pts_cir)
"""


"""
#voiles
X1,Y1,Z1=carreau(pts_reg)
#tracage(X1,Y1,Z1,pts_reg)
rotation_Y_Z(X1,Y1,Z1,pts_reg)
#rotation_X_Z(X1,Y1,Z1,pts_reg)


translation_ez(pts_reg,4)
X1,Y1,Z1=carreau(pts_reg)

rotation_Y_Z(X1,Y1,Z1,pts_reg)

"""


##rectangles
"""
pts_c_4x3=[[[0,0,0],[5,0,10],[10,0,0],[14,0,2]],
       [[0,2,2],[5,2,15],[10,2,5],[14,2,2]],
       [[0,6,0],[5,6,7],[10,6,1],[14,6,2]],
       [[0,10,1],[5,10,9],[10,10,10],[14,10,15]]]


pts_c_3x4=[[[0,0,0],[0,2,2],[0,6,0]],
            [[5,0,10],[5,2,15],[5,6,7]],
            [[10,0,0],[10,2,5],[10,6,1]],
            [[14,0,2],[14,2,2],[14,6,2]]]
"""

"""
pts_3x4=np.array(pts_c_4x3)

X4,Y4,Z4=carreau(pts_3x4)

tracage(X4,Y4,Z4,pts_3x4)


"""

"""
pts_c_nxm=[[[00,0,0],[5,0,10],[10,0,0],[14,0,2]],
       [[0,2,2],[5,2,15],[10,2,5],[14,2,2]],
       [[0,6,0],[5,6,7],[10,6,1],[14,6,2]],
       [[0,10,1],[5,10,9],[10,10,10],[14,10,15]],
       [[0,5,-3],[5,5,-1],[10,5,-4],[14,5,0]]]

pts_nxm=np.array(pts_c_nxm)

X4,Y4,Z4=carreau(pts_nxm)

tracage(X4,Y4,Z4,pts_nxm)
"""

"""
teste_repere=[[[5,-5,-2],[6,-1,-40],[10,-5,0]],
              [[0,0,5],[6,0,10],[14,0,6]],
              [[0,3,9],[6,3,16],[14,3,8]],
              [[0,7,5],[6,7,9],[14,7,5]]]

tr=np.array(teste_repere)
print(tr)
X4,Y4,Z4=carreau(tr)

tracage(X4,Y4,Z4,tr)

"""
##debut u nuage

def afinale(pts):

    X,Y,Z = carreau(pts)
    tracage(X,Y,Z,pts)

    return()

f_g_hl=[[[0,0,0],[2,0,0],[11,2,0]],
       [[0,3,0],[2,3,7],[11,3,0]],
       [[0,6,0],[2,6,0],[11,5,0]]]



f_g_hl2=[[[0,10,0],[2,10,0],[11,12,0]],
       [[0,13,0],[2,13,7],[11,13,0]],
       [[0,16,0],[2,16,0],[11,15,0]]]



f_g_h=np.array(f_g_hl)

f_g_h5=np.array(f_g_hl2)


f_g_h1 = [[[2,0,0],[11,2,0]],
       [[2,3,7],[11,3,0]],
       [[2,6,0],[11,5,0]]]


f_g_h2 = [[[0,0,0],[2,0,0]],
            [[0,3,0],[2,3,7]],
            [[0,6,0],[2,6,0]]]




## le nuage enfin svp

## les points du nuage


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
p46=np.array([61,9.5,0])
p47=np.array([62.5,9.5,0])





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
pf72=(p25+p55)/2 + np.array([1.5,0,2])



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

#Bizarre ya 2 p42

f9 = [[p51,p42,p42,p52],
        [pf90,pf91,pf92,pf93],
        [p61,p62,p63,p64]]


""" figure 10"""

pf104 = [2,3,0]
pf105 = [2.5,3,0]

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

pf124 = [28.3,3,0]
pf125 = [34.6,3,0]

pf120=(p63+p83)/2
pf121=(p64+pf124)/2 + np.array([0,0,4])
pf122 = (p65+pf125)/2 + np.array([0,0,4])
pf123 = (p66+p84)/2



f12 = [[p63,p64,p65,p66],
        [pf120,pf121,pf122,pf123],
        [p83,pf124,pf125,p84]]


""" figure 13"""

pf134 = [48,3,0]
pf135 = [54,3,0]

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

pt10 = [8.75,17,0]
pt11 = [14.5,17,0]


t1 = [[p01,pt10,pt11,p02],
        [p11,p12,p13,p1b]]



""" figure top 2 """

pt20 = [28.25,17,0]
pt21 = [34.75,17,0]


t2 = [[p03,pt20,pt21,p04],
        [p1b2,p14,p15,p16]]

""" figure top 3 """



pt30 = [48.5,17,0]
pt31 = [54.25,17,0]


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

print(j0)
def trace_list(f):
    for i in range(0,len(f)):

        fi = np.array(f[i])
        X, Y, Z = carreau(fi)
        rotation_Y_Z(X,Y,Z, fi)
    return ()


trace_list(ftab)
trace_list(jtab)
trace_list(ttab)
ax.scatter(60, 15,15)


##Vue de face (petite face)

q00=np.array([1.5,17,-21])
q01=np.array([1.5,17,-19.5])
q02=np.array([1.5,17,-1.5])
q03=np.array([1.5,17,0])


q10=np.array([1.5,16,-21])
q11=np.array([1.5,16,-19.5])
q12=np.array([1.5,16,-1.5])
q13=np.array([1.5,16,0])

q20=np.array([1.5,13.6,-11])
q21=np.array([1.5,13.6,-10])
q22=np.array([1.5,13.3,-11])
q23=np.array([1.5,13.3,-10])

q30=np.array([1.5,10.5,-21])
q31=np.array([1.5,10.5,-19.5])
q32=np.array([1.5,10.5,-1.5])
q33=np.array([1.5,10.5,0])

q40=np.array([1.5,9.5,-21])
q41=np.array([1.5,9.5,-19.5])
q42=np.array([1.5,9.5,-1.5])
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

qp10 =(0,13,1)
qp11=(q12+q32)/2 + np.array([-4,0,0])
qp12 = (q21+q23)/2


fp2 = [[q21,q12,q13],
        [qp12,qp11,qp10],
        [q23,q32,q33]]




















"""
afinale(np.array(f1))

afinale(np.array(f2))
afinale(np.array(f3))
afinale(np.array(f4))
afinale(np.array(f5))
afinale(np.array(f6))
afinale(np.array(f7))
"""
"""


afinale(np.array(f1))

afinale(np.array(f2))
"""
"""X,Y,Z = carreau(np.array(f1))
rotation_X_Z(X,Y,Z,np.array(f1))
"""
"""
X,Y,Z = carreau(np.array(f1))
rotation_Y_Z(X,Y,Z,np.array(f1))



#suite
X,Y,Z = carreau(np.array(f2))
rotation_Y_Z(X,Y,Z,np.array(f2))





X,Y,Z = carreau(np.array(f3))
rotation_Y_Z(X,Y,Z,np.array(f3))

X,Y,Z = carreau(np.array(f4))
rotation_Y_Z(X,Y,Z,np.array(f4))

X,Y,Z = carreau(np.array(f5))
rotation_Y_Z(X,Y,Z,np.array(f5))

X,Y,Z = carreau(np.array(f6))
rotation_Y_Z(X,Y,Z,np.array(f6))

X,Y,Z = carreau(np.array(f7))
rotation_Y_Z(X,Y,Z,np.array(f7))
"""
#"racordement de surface, maniere de ponderer des points, surface reglé, surface developable, b spline module scy pi de pyhton, different temps"




##Autre coté

esp = 8

"""
f1_rev = np.array(f1)
translation_ez(f1_rev,esp)
X,Y,Z = carreau(f1_rev)
rotation_Y_Z_inv(X,Y,Z,f1_rev)

f2_rev = np.array(f2)
translation_ez(f2_rev,esp)
X,Y,Z = carreau(f2_rev)
rotation_Y_Z_inv(X,Y,Z,f2_rev)

f3_rev = np.array(f3)
translation_ez(f3_rev,esp)
X,Y,Z = carreau(f3_rev)
rotation_Y_Z_inv(X,Y,Z,f3_rev)


f4_rev = np.array(f4)
translation_ez(f4_rev,esp)
X,Y,Z = carreau(f4_rev)
rotation_Y_Z_inv(X,Y,Z,f4_rev)


f5_rev = np.array(f5)
translation_ez(f5_rev,esp)
X,Y,Z = carreau(f5_rev)
rotation_Y_Z_inv(X,Y,Z,f5_rev)


f6_rev = np.array(f6)
translation_ez(f6_rev,esp)
X,Y,Z = carreau(f6_rev)
rotation_Y_Z_inv(X,Y,Z,f6_rev)


f7_rev = np.array(f7)
translation_ez(f7_rev,esp)
X,Y,Z = carreau(f7_rev)
rotation_Y_Z_inv(X,Y,Z,f7_rev)

"""




##tube



plt.show()


