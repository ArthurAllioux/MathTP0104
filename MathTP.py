import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as st
import numpy as np

df = pd.read_excel('MeteoriteDB.xlsx', sheet_name = "Sheet1") # Allow me to read the sheet named "Sheet1" of MeteoriteDB.xlsx
df.dropna(inplace = True) # Deletes all the lines that are incomplete
#print(df)

#Question N°1 : Make an histogram of the mass distribution of meteorites. 
# Do it again for the meteorites having a mass less or equal to 50 000 grams.

Mass = []
#len(df.index)
for i in range(100):
    cell = df.iloc[i,4]
    for j, char in enumerate(cell):
        if char == "," :
            fin = j
    mass = float(cell[9:fin])
    Mass.append(mass)

plt.hist(Mass, bins = 40)
plt.xlabel("Values")
plt.ylabel("Number of events")
plt.title("Histogram of the mass ditribution")
#plt.show()
Mass_50000 = []
for i in range(len(Mass)):
    if Mass[i] <= 50000:
        Mass_50000.append(Mass[i])

plt.hist(Mass_50000, bins = 100)
plt.xlabel("Values")
plt.ylabel("Number of events")
plt.title("Histogram of the mass ditribution of meteorites having a mass less or equal to 50000 grammes")
#plt.show()

#Question N°2 : Make a plot of the number of meteorites as a function of time (by year). 
# Find a linear fit (y = ax + b) that approximates the trend of the curve. 
# Using this function say what would be the number of landing meteorites next year. 
# Is this approach of prediction scientifically robust?


Years = []
for i in range(len(df.index)):
    cell = df.iloc[i,6]
    if isinstance(cell, str) == True:
        Years.append(int(cell))
    else:
        Years.append(cell.year)

dictionary = {}
for year in Years:
    number_by_year = Years.count(year)
    dictionary[year] = number_by_year
#print(dictionary)
number_of_meteorites = dictionary.items()
number_of_meteorites = sorted(number_of_meteorites)
x, y = zip(*number_of_meteorites)



plt.plot(x, y)
plt.xlim(1850,2021)
plt.xlabel('Year')
plt.ylabel('Number of meteorites')
plt.title('Number of meteorites by year')
#plt.show()

plt.plot(x, y)
plt.xlim(1960,2014)
plt.xlabel('Year')
plt.ylabel('Number of meteorites')
plt.title('Number of meteorites by year')
#plt.show()

def linearfit(x,a,b):
    return a * x + b

popt, _ = curve_fit(linearfit,x,y)
a,b = popt
print('y = %.5f * x + %.5f' % (a,b))

print(linearfit(2022,a,b))

#Question N°3 : We will concentrate now in the case of Oman.
# Create a plot of this country with different points representing the spatial distribution of the landing sites.

df_coord = df['Coordinates']
dico_Coord = {}
for i in range(len(df.index)):
    cell = df.iloc[i,7]
    for j, char in enumerate(cell):
        if char == "{":
            debut_x = j+1
    for k, char in enumerate(cell):
        if char == "," :
            fin_x = k
            debut_y = k+2
    for l, char in enumerate(cell):
        if char == "}" :
            fin_y = l
    x = float(cell[debut_x:fin_x])
    y = float(cell[debut_y:fin_y])
    dico_Coord[x] = y
#print(dico_Coord)
Coord_Oman = {}
for x,y in dico_Coord.items():
    if x > 16 and x < 25:
        if y > 52 and y < 60:
            Coord_Oman[x] = y
#print(Coord_Oman)

world = dico_Coord.items()
world = sorted(world)
y, x = zip(*world)

plt.scatter(x,y,c = '#2000DB')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Representation of landing sites in the world')
#plt.show()


oman = Coord_Oman.items()
oman = sorted(oman)
y, x = zip(*oman)

plt.scatter(x,y,c = '#4500DB')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Representation of landing sites in Oman')
#plt.show()

#Question N°4 : Propose a distribution (uniform, gaussian, cobinaison of different ones...)
# in order to describe the distribution of meteorite landing sites in Oman.

# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - deltaX
xmax = max(x) + deltaX
ymin = min(y) - deltaY
ymax = max(y) + deltaY
print(xmin, xmax, ymin, ymax)
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)

fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='jet')
ax.imshow(np.rot90(f), cmap='jet', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('2D Gaussian Kernel density estimation')

plt.figure(figsize=(8,8))
for j in range(len(cset.allsegs)):
    for ii, seg in enumerate(cset.allsegs[j]):
        plt.plot(seg[:,0], seg[:,1], '.-', label=f'Cluster{j}, level{ii}')
plt.legend()

h =plt.hist2d(x, y)
plt.colorbar(h[3])


#Question N°5 : Based on that distribution compute the probability that a meteorite land in the circle of center
# {latitude,longitude}={18.9644, 53.9555} and radius equals to 100 Kilometers.
                    
latitude,longitude=18.9644, 53.9555
Coord_circle = {}                   
for x,y in dico_Coord.items():
    if x > latitude-1 and x < latitude+1:
        if y > longitude-1 and y < longitude+1:
            Coord_circle[x] = y 

oman = Coord_Oman.items()
oman = sorted(oman)
y, x = zip(*oman)
taux1=len(oman)

circle = Coord_circle.items()
circle = sorted(circle)
y1, x1 = zip(*circle)
taux2 = len(circle)
print(taux2/taux1)


plt.scatter(x,y,c = '#4500DB')
plt.scatter(x1,y1,c = '#17becf')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Representation of landing sites in Oman')
plt.show()

