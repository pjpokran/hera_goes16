#!/home/poker/miniconda3/bin/python

import netCDF4
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
#import os.rename
#import os.remove
import shutil
import sys

band="09"
filechar=['AA','AB','AC','AD','AE','AF','AG','AH','AI','AJ','AK','AL','AM',
          'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
          'BA','BB','BC','BD','BE','BF','BG','BH','BI','BJ','BK','BL','BM',
          'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ']

#print(filechar[1])

prod_id = "TIRS"
#dt="201703051957"
dt = sys.argv[1]

#f = netCDF4.Dataset("/weather/data/goes16/TIRC/"+band+"/"+dt+"_PAA.nc")
f = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_PAA.nc")
a = np.zeros(shape=(f.product_rows,f.product_columns))
xa= np.zeros(shape=(f.product_columns))
ya= np.zeros(shape=(f.product_rows))


print(f)

data_var = f.variables['Sectorized_CMI']
a[0:f.product_tile_height,0:f.product_tile_width] = data_var[:]
#data_var2 = g.variables['Sectorized_CMI']

print(data_var)

x = f.variables['x'][:]
y = f.variables['y'][:]
xa[f.tile_column_offset:f.tile_column_offset+f.product_tile_width] = x[:]
ya[f.tile_row_offset:f.tile_row_offset+f.product_tile_height] = y[:]

if f.number_product_tiles > 1:
# this goes from 1 to number of tiles - 1
    for i in range(1,f.number_product_tiles):
#    print(filechar[i])
        if os.path.isfile("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc"):
            g = netCDF4.Dataset("/weather/data/goes16/"+prod_id+"/"+band+"/"+dt+"_P"+filechar[i]+".nc")
#        print(g)
            data_var2 = g.variables['Sectorized_CMI']
            a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
            x = g.variables['x'][:]
            y = g.variables['y'][:]
            xa[g.tile_column_offset:g.tile_column_offset+g.product_tile_width] = x[:]
            ya[g.tile_row_offset:g.tile_row_offset+g.product_tile_height] = y[:]
            g.close


#a[g.tile_column_offset:g.tile_column_offset+g.product_tile_width,g.tile_row_offset:g.tile_row_offset+g.product_tile_height]=data_var[:]
    a[g.tile_row_offset:g.tile_row_offset+g.product_tile_height,g.tile_column_offset:g.tile_column_offset+g.product_tile_width]=data_var2[:]
#print(a)

# swap zeros for ones
a[a==0.] = 1.

#print(np.average(a))
#if np.average(a) > 0.75:
#    quit()


proj_var = f.variables[data_var.grid_mapping]

xa=xa*proj_var.perspective_point_height/1000000.
ya=ya*proj_var.perspective_point_height/1000000.

import cartopy.crs as ccrs

##########################################################
## HACK for elipse and mask being drawn inside of the actual edge of the earth
import shapely.geometry as sgeom
def __init__(self, projection, satellite_height=35785831,
             central_longitude=0.0, central_latitude=0.0,
             false_easting=0, false_northing=0, globe=None):
    proj4_params = [('proj', projection), ('lon_0', central_longitude),
                    ('lat_0', central_latitude), ('h', satellite_height),
                    ('x_0', false_easting), ('y_0', false_northing),
                    ('units', 'm')]
    super(ccrs._Satellite, self).__init__(proj4_params, globe=globe)

    # TODO: Let the globe return the semimajor axis always.
    a = np.float(self.globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS)
    b = np.float(self.globe.semiminor_axis or a)
    h = np.float(satellite_height)
    max_x = h * np.arcsin(a / (a + h))
    max_y = h * np.arcsin(b / (b + h))

    coords = ccrs._ellipse_boundary(max_x, max_y,
                                    false_easting, false_northing, 61)
    self._boundary = sgeom.LinearRing(coords.T)
    self._xlim = self._boundary.bounds[::2]
    self._ylim = self._boundary.bounds[1::2]
    self._threshold = np.diff(self._xlim)[0] * 0.02

ccrs._Satellite.__init__ = __init__
##########################################################
##########################################################
## HACK for elipse and mask being drawn inside of the actual edge of the earth
#import math
#import shapely.geometry as sgeom
#def override_ellipse(self):
#    a = np.float(self.globe.semimajor_axis)
#    b = np.float(self.globe.semiminor_axis or a)
#    h = np.float(35785831.0)
#    max_x = 1.011 * h * math.atan(a / (a + h))
#    max_y = 1.011 * h * math.atan(b / (a + h))
#    coords = ccrs._ellipse_boundary(max_x, max_y, 0, 0, 61)
#    self._boundary = sgeom.LinearRing(coords.T)
#    self._xlim = self._boundary.bounds[::2]
#    self._ylim = self._boundary.bounds[1::2]
#    self._threshold = np.diff(self._xlim)[0] * 0.02
##########################################################


# Create a Globe specifying a spherical earth with the correct radius
globe = ccrs.Globe(semimajor_axis=proj_var.semi_major,
                   semiminor_axis=proj_var.semi_minor)

#proj = ccrs.LambertConformal(central_longitude=proj_var.longitude_of_central_meridian,
#                             central_latitude=proj_var.latitude_of_projection_origin,
#                             standard_parallels=[proj_var.standard_parallel],
#                             globe=globe)

proj = ccrs.Geostationary(central_longitude=-75.0, 
                          satellite_height=35785831, 
                          globe=globe)

##########################################################
## HACK for elipse and mask being drawn inside of the actual edge of the earth
#override_ellipse(proj)
##########################################################


image_rows=f.product_rows
image_columns=f.product_columns
namer_image_crop_top=0
namer_image_crop_bottom=-800
namer_image_crop_left=300
namer_image_crop_right=-300

namer_image_size_y=(image_rows+namer_image_crop_bottom-namer_image_crop_top)
namer_image_size_x=(image_columns+namer_image_crop_right-namer_image_crop_left)

print("namer image size")
print(namer_image_size_x, namer_image_size_y)

#wi_image_size_x=float(wi_image_size_x)/120.
#wi_image_size_y=float(wi_image_size_y)/120.
namer_image_size_x=float(namer_image_size_x)/80.
namer_image_size_y=float(namer_image_size_y)/80.

#mw_image_crop_top=250
#mw_image_crop_bottom=-3200
#mw_image_crop_left=1750
#mw_image_crop_right=-2150
#
#mw_image_size_y=(image_rows+mw_image_crop_bottom-mw_image_crop_top)
#mw_image_size_x=(image_columns+mw_image_crop_right-mw_image_crop_left)
#
#print("mw image size")
#print(mw_image_size_x, mw_image_size_y)
#
#mw_image_size_x=float(mw_image_size_x)/150.
#mw_image_size_y=float(mw_image_size_y)/150.
#
#conus_image_crop_top=100
#conus_image_crop_bottom=-1800
#conus_image_crop_left=100
#conus_image_crop_right=-450
#
#conus_image_size_y=(image_rows+conus_image_crop_bottom-conus_image_crop_top)
#conus_image_size_x=(image_columns+conus_image_crop_right-conus_image_crop_left)
#
#print("conus image size")
#print(conus_image_size_x, conus_image_size_y)
#
#conus_image_size_x=float(conus_image_size_x)/300.
#conus_image_size_y=float(conus_image_size_y)/300.

# Create a new figure with size 10" by 10"
fig = plt.figure(figsize=(namer_image_size_x,namer_image_size_y),dpi=80.)
fig2 = plt.figure(figsize=(image_columns/160.,image_rows/160.),dpi=160.)
#fig3 = plt.figure(figsize=(conus_image_size_x,conus_image_size_y),dpi=160.)
#fig2 = plt.figure(figsize=(image_columns/200.,image_rows/200.))
fig9 = plt.figure(figsize=(image_columns/78.,image_rows/78.))

# Put a single axes on this figure; set the projection for the axes to be our
# Lambert conformal projection
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
#ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
ax9 = fig9.add_subplot(1, 1, 1, projection=proj)


cdict = {'red': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                 (0.29, 1.00, 1.00),
                 (0.61, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

cdict2 = {'red': ((0.0, 0.0, 0.0),
                 (0.208, 0.0, 0.0),
                 (0.379, 1.0, 1.0),
                 (0.483, 0.0, 0.0),
                 (0.572, 1.0, 1.0),
                 (0.667, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
         'green': ((0.0, 0.0, 0.0),
                   (0.208, .423, .423),
                   (0.379, 1.0, 1.0),
                   (0.483, 0.0, 0.0),
                   (0.572, 1.0, 1.0),
                   (0.667, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 0.0, 0.0),
                  (0.208, 0.0, 0.0),
                  (0.379, 1.0, 1.0),
                  (0.483,0.651, 0.651),
                  (0.572, 0.0, 0.0),
                  (0.667, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}


import matplotlib as mpl

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)


# Plot the data using a simple greyscale colormap (with black for low values);
# set the colormap to extend over a range of values from 140 to 255.
# Note, we save the image returned by imshow for later...
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#               cmap='Greys_r', norm=plt.Normalize(0, 256))
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper',
#im = ax.imshow(a[:], extent=(xa[0], xa[-1], ya[-1], ya[0]), origin='upper',
#               cmap='Greys_r')
#im = ax.imshow(a[250:-3000,2000:-2000], extent=(xa[2000],xa[-2000],ya[-3000],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(a[250:-3500,2500:-2200], extent=(xa[2500],xa[-2200],ya[-3500],ya[250]), origin='upper',cmap='Greys_r')
#im = ax.imshow(data_var[:], extent=(x[0], x[-1], y[0], y[-1]), origin='upper')
#im = ax2.imshow(a[:], extent=(xa[1],xa[-1],ya[-1],ya[1]), origin='upper', cmap='Greys_r')

im = ax.imshow(a[namer_image_crop_top:namer_image_crop_bottom,namer_image_crop_left:namer_image_crop_right], extent=(xa[namer_image_crop_left],xa[namer_image_crop_right],ya[namer_image_crop_bottom],ya[namer_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.)
#im = ax2.imshow(a[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right], extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.)
im = ax2.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.)
#im = ax3.imshow(a[conus_image_crop_top:conus_image_crop_bottom,conus_image_crop_left:conus_image_crop_right], extent=(xa[conus_image_crop_left],xa[conus_image_crop_right],ya[conus_image_crop_bottom],ya[conus_image_crop_top]), origin='upper',cmap=my_cmap, vmin=162., vmax=330.)
im = ax9.imshow(a[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper', cmap=my_cmap, vmin=162., vmax=330.)

im = ax.coastlines(resolution='50m', color='green')
im = ax2.coastlines(resolution='50m', color='green')
im = ax9.coastlines(resolution='50m', color='green')

import cartopy.feature as cfeat

# Add country borders with a thick line.
ax.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
#ax3.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')

# Set up a feature for the state/province lines. Tell cartopy not to fill in the polygons
state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

# Add the feature with dotted lines, denoted by ':'
ax.add_feature(state_boundaries, linestyle=':')
ax2.add_feature(state_boundaries, linestyle=':')
#ax3.add_feature(state_boundaries, linestyle=':')
ax9.add_feature(state_boundaries, linestyle=':')

# Redisplay modified figure
#fig
#fig2

import datetime

time_var = f.start_date_time

jyr = time_var[0:4]
jday = time_var[4:7]
#print(jday)

date = datetime.datetime(int(jyr), 1, 1) + datetime.timedelta(int(jday)-1)

time_string = 'GOES-16 Band 9 mid-level water vapor valid %s '%date.strftime('%Y %b %d')+time_var[7:9]+":"+time_var[9:11]+":"+time_var[11:13]+" GMT"
print(time_string)

from matplotlib import patheffects
outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

#2017/065 20:04:00:30
text = ax.text(0.50, 0.03, time_string,
    horizontalalignment='center', transform = ax.transAxes,
    color='yellow', fontsize='large', weight='bold')

text.set_path_effects(outline_effect)

text2 = ax2.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax2.transAxes,
    color='yellow', fontsize='large', weight='bold')

text2.set_path_effects(outline_effect)

#text = ax3.text(0.50, 0.90, time_string,
#    horizontalalignment='center', transform = ax3.transAxes,
#    color='yellow', fontsize='large', weight='bold')

text9 = ax9.text(0.50, 0.97, time_string,
    horizontalalignment='center', transform = ax9.transAxes,
    color='yellow', fontsize='large', weight='bold')

text9.set_path_effects(outline_effect)



filename1="/whirlwind/goes16/wv/namer/"+dt+"_namer.jpg"
filename2="/whirlwind/goes16/wv/fulldisk/"+dt+"_fulldisk.jpg"
#filename3="/whirlwind/goes16/wv/conus/"+dt+"_conus.jpg"
filename9="/whirlwind/goes16/wv/fulldisk_full/"+dt+"_fulldisk_full.jpg"

fig.savefig(filename1, bbox_inches='tight')
fig2.savefig(filename2, bbox_inches='tight')
#fig2.savefig(filename2jpg, bbox_inches='tight')
#fig3.savefig(filename3, bbox_inches='tight')
fig9.savefig(filename9, bbox_inches='tight')

#quit()

#import os.rename    # os.rename(src,dest)
#import os.remove    # os.remove path
#import shutil.copy  # shutil.copy(src, dest)

os.remove("/whirlwind/goes16/wv/namer/latest_namer_24.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_23.jpg", "/whirlwind/goes16/wv/namer/latest_namer_24.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_22.jpg", "/whirlwind/goes16/wv/namer/latest_namer_23.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_21.jpg", "/whirlwind/goes16/wv/namer/latest_namer_22.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_20.jpg", "/whirlwind/goes16/wv/namer/latest_namer_21.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_19.jpg", "/whirlwind/goes16/wv/namer/latest_namer_20.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_18.jpg", "/whirlwind/goes16/wv/namer/latest_namer_19.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_17.jpg", "/whirlwind/goes16/wv/namer/latest_namer_18.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_16.jpg", "/whirlwind/goes16/wv/namer/latest_namer_17.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_15.jpg", "/whirlwind/goes16/wv/namer/latest_namer_16.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_14.jpg", "/whirlwind/goes16/wv/namer/latest_namer_15.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_13.jpg", "/whirlwind/goes16/wv/namer/latest_namer_14.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_12.jpg", "/whirlwind/goes16/wv/namer/latest_namer_13.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_11.jpg", "/whirlwind/goes16/wv/namer/latest_namer_12.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_10.jpg", "/whirlwind/goes16/wv/namer/latest_namer_11.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_9.jpg", "/whirlwind/goes16/wv/namer/latest_namer_10.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_8.jpg", "/whirlwind/goes16/wv/namer/latest_namer_9.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_7.jpg", "/whirlwind/goes16/wv/namer/latest_namer_8.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_6.jpg", "/whirlwind/goes16/wv/namer/latest_namer_7.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_5.jpg", "/whirlwind/goes16/wv/namer/latest_namer_6.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_4.jpg", "/whirlwind/goes16/wv/namer/latest_namer_5.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_3.jpg", "/whirlwind/goes16/wv/namer/latest_namer_4.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_2.jpg", "/whirlwind/goes16/wv/namer/latest_namer_3.jpg")
os.rename("/whirlwind/goes16/wv/namer/latest_namer_1.jpg", "/whirlwind/goes16/wv/namer/latest_namer_2.jpg")

shutil.copy(filename1, "/whirlwind/goes16/wv/namer/latest_namer_1.jpg")


os.remove("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_24.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_23.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_24.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_22.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_23.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_21.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_22.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_20.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_21.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_19.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_20.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_18.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_19.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_17.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_18.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_16.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_17.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_15.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_16.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_14.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_15.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_13.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_14.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_12.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_13.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_11.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_12.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_10.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_11.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_9.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_10.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_8.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_9.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_7.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_8.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_6.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_7.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_5.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_6.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_4.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_5.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_3.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_4.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_2.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_3.jpg")
os.rename("/whirlwind/goes16/wv/fulldisk/latest_fulldisk_1.jpg", "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_2.jpg")

shutil.copy(filename2, "/whirlwind/goes16/wv/fulldisk/latest_fulldisk_1.jpg")


shutil.copy(filename9, "/whirlwind/goes16/wv/fulldisk_full/latest_fulldisk_full_1.jpg")
